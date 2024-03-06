#include "mpi_label_propagation.h"

#include <kaminpar-common/datastructures/marker.h>

#include <numeric>

namespace kaminpar::dist {
MpiLP::MpiLP(Context const& ctx) : BaseLP(ctx) {}

void MpiLP::enforce_max_cluster_weights(NodeID const from, NodeID const to) {
  int size;
  MPI_Comm_size(_graph->communicator(), &size);

  // Step 1: Aggregate the weight deltas

  _cluster_weight_deltas.clear();
  std::vector<int> sendcounts(size);
  for (NodeID const u : _graph->nodes(from, to)) {
    GlobalNodeID const old_gcluster = _prev_clustering[u];
    GlobalNodeID const new_gcluster = _clustering[u];

    if (old_gcluster != new_gcluster) {
      NodeWeight const w_u = _graph->node_weight(u);

      if (!_graph->is_owned_global_node(old_gcluster)) {
        if (_cluster_weight_deltas.find(old_gcluster) ==
            _cluster_weight_deltas.end()) {
          const PEID owner = _graph->find_owner_of_global_node(old_gcluster);
          sendcounts[owner] += 2;
        }
        _cluster_weight_deltas[old_gcluster] -= w_u;
      }

      if (!_graph->is_owned_global_node(new_gcluster)) {
        if (_cluster_weight_deltas.find(new_gcluster) ==
            _cluster_weight_deltas.end()) {
          const PEID owner = _graph->find_owner_of_global_node(new_gcluster);
          sendcounts[owner] += 2;
        }
        _cluster_weight_deltas[new_gcluster] += w_u;
      }
    }
  }

  std::vector<int> sdispls(size + 1);
  std::exclusive_scan(sendcounts.begin(), sendcounts.end(), sdispls.begin() + 1,
                      0);
  std::vector<std::int64_t> sendbuf(sdispls.back() + sendcounts.back());

  for (auto const& [gcluster, delta] : _cluster_weight_deltas) {
    const PEID pe = _graph->find_owner_of_global_node(gcluster);
    sendbuf[sdispls[pe + 1]++] = gcluster;
    sendbuf[sdispls[pe + 1]++] = delta;
  }

  std::vector<int> recvcounts(size);
  std::vector<int> rdispls(size + 1);

  MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT,
               _graph->communicator());
  std::inclusive_scan(recvcounts.begin(), recvcounts.end(),
                      rdispls.begin() + 1);

  std::vector<std::int64_t> recvbuf(rdispls.back());

  MPI_Alltoallv(sendbuf.data(), sendcounts.data(), sdispls.data(), MPI_INT64_T,
                recvbuf.data(), recvcounts.data(), rdispls.data(), MPI_INT64_T,
                _graph->communicator());

  for (std::size_t i = 0; i < recvbuf.size();) {
    std::uint64_t const gcluster = recvbuf[i++];
    std::int64_t const delta = recvbuf[i++];

    change_cluster_weight(gcluster, delta, false);
  }

  // Step 2: reply with the new global cluster weights
  for (std::size_t i = 0; i < recvbuf.size();) {
    std::uint64_t const gcluster = recvbuf[i++];
    recvbuf[i++] = cluster_weight(gcluster);
  }

  MPI_Alltoallv(recvbuf.data(), recvcounts.data(), rdispls.data(), MPI_INT64_T,
                sendbuf.data(), sendcounts.data(), sdispls.data(), MPI_INT64_T,
                _graph->communicator());

  // Step 3: Update cluster weight and revert moves if necessary

  bool violates_max_cluster_weight = false;
  for (std::size_t i = 0; i < sendbuf.size();) {
    GlobalNodeID const gcluster = sendbuf[i++];
    GlobalNodeWeight new_w_gcluster = sendbuf[i++];
    GlobalNodeID const old_w_gcluster = cluster_weight(gcluster);

    if (new_w_gcluster > _max_cluster_weight) {
      violates_max_cluster_weight = true;

      auto increase_by_others = new_w_gcluster - old_w_gcluster;
      auto const increase_by_me = _cluster_weight_deltas[gcluster];

      new_w_gcluster =
          _max_cluster_weight +
          (1.0 * increase_by_me / (increase_by_others + increase_by_me)) *
              (new_w_gcluster - _max_cluster_weight);
    }

    change_cluster_weight(gcluster, -old_w_gcluster + new_w_gcluster, true);
  }

  if (violates_max_cluster_weight) {
    for (NodeID const u : _graph->nodes(from, to)) {
      GlobalNodeID const old_gcluster = _prev_clustering[u];
      GlobalNodeID const new_gcluster = _clustering[u];

      if (cluster_weight(new_gcluster) > _max_cluster_weight) {
        move_node(u, _graph->node_weight(u), new_gcluster, old_gcluster);
      }
    }
  }
}

void MpiLP::synchronize_ghost_node_clusters(NodeID const from,
                                            NodeID const to) {
  int size;
  MPI_Comm_size(_graph->communicator(), &size);

  std::vector<std::vector<std::uint64_t>> sendbox(size);

  Marker created_message_for_pe(size);
  for (NodeID u = from; u < to; ++u) {
    if (_prev_clustering[u] != _clustering[u]) {
      for (auto const& [e, v] : _graph->neighbors(u)) {
        if (_graph->is_owned_node(v)) {
          continue;
        }

        PEID const pe = _graph->ghost_owner(v);
        if (!created_message_for_pe.get(pe)) {
          sendbox[pe].push_back(_graph->local_to_global_node(u));
          sendbox[pe].push_back(_clustering[u]);
          created_message_for_pe.set(pe);
        }
      }

      created_message_for_pe.reset();
    }
  }

  std::vector<int> sendcounts(size);
  std::vector<int> sdispls(size + 1);
  for (int pe = 0; pe < size; ++pe) {
    sendcounts[pe] = asserting_cast<int>(sendbox[pe].size());
    sdispls[pe + 1] = sdispls[pe] + sendcounts[pe];
  }

  std::vector<std::uint64_t> sendbuf(sdispls.back());
  for (int pe = 0; pe < size; ++pe) {
    std::copy(sendbox[pe].begin(), sendbox[pe].end(),
              sendbuf.begin() + sdispls[pe]);
  }

  std::vector<int> recvcounts(size);
  MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT,
               _graph->communicator());

  std::vector<int> rdispls(size + 1);
  std::inclusive_scan(recvcounts.begin(), recvcounts.end(),
                      rdispls.begin() + 1);

  std::vector<std::uint64_t> recvbuf(rdispls.back());

  MPI_Alltoallv(sendbuf.data(), sendcounts.data(), sdispls.data(), MPI_UINT64_T,
                recvbuf.data(), recvcounts.data(), rdispls.data(), MPI_UINT64_T,
                _graph->communicator());

  for (std::size_t i = 0; i < recvbuf.size();) {
    GlobalNodeID const gnode = recvbuf[i++];
    GlobalNodeID const new_gcluster = recvbuf[i++];

    NodeID const lnode = _graph->global_to_local_node(gnode);

    _clustering[lnode] = new_gcluster;

    // Update the cluster weights if we do not already have the exact cluster
    // weight due to enforce_max_cluster_weights(): (i) if we own the clusters,
    // we have its exact weight (ii) if we have the cluster in our
    // _cluster_weight_deltas, we have its exact weight In all other cases, we
    // do not know its exact weight: track the delta
    NodeWeight const w_lnode = _graph->node_weight(lnode);
    GlobalNodeID const old_gcluster = _prev_clustering[lnode];

    if (!_graph->is_owned_global_node(old_gcluster) &&
        _cluster_weight_deltas.find(old_gcluster) ==
            _cluster_weight_deltas.end()) {
      change_cluster_weight(old_gcluster, -w_lnode, true);
    }
    if (!_graph->is_owned_global_node(new_gcluster) &&
        _cluster_weight_deltas.find(new_gcluster) ==
            _cluster_weight_deltas.end()) {
      change_cluster_weight(new_gcluster, w_lnode, false);
    }

    _prev_clustering[lnode] = new_gcluster;
  }
}
}  // namespace kaminpar::dist

