#include "kamping_dispatch_label_propagation.h"

#include <kaminpar-common/datastructures/marker.h>

#include <kamping/plugin/alltoall_dispatch.hpp>
#include <kamping/plugin/alltoall_grid.hpp>
#include <kamping/utils/flatten.hpp>

namespace kaminpar::dist {
using namespace ::kamping;

using Communicator =
    Communicator<std::vector, ::kamping::plugin::GridCommunicator,
                 ::kamping::plugin::DispatchAlltoall>;

KampingDispatchLP::KampingDispatchLP(Context const& ctx) : BaseLP(ctx) {}

void KampingDispatchLP::enforce_max_cluster_weights(NodeID const from,
                                               NodeID const to) {
  Communicator comm(_graph->communicator());

  // Step 1: Aggregate the weight deltas

  _cluster_weight_deltas.clear();
  std::vector<int> sendcounts(comm.size());
  for (NodeID const u : _graph->nodes(from, to)) {
    GlobalNodeID const old_gcluster = _prev_clustering[u];
    GlobalNodeID const new_gcluster = _clustering[u];

    if (old_gcluster != new_gcluster) {
      NodeWeight const w_u = _graph->node_weight(u);

      if (!_graph->is_owned_global_node(old_gcluster)) {
        if (_cluster_weight_deltas.find(old_gcluster) ==
            _cluster_weight_deltas.end()) {
          const PEID owner = _graph->find_owner_of_global_node(old_gcluster);
          ++sendcounts[owner];
        }
        _cluster_weight_deltas[old_gcluster] -= w_u;
      }

      if (!_graph->is_owned_global_node(new_gcluster)) {
        if (_cluster_weight_deltas.find(new_gcluster) ==
            _cluster_weight_deltas.end()) {
          const PEID owner = _graph->find_owner_of_global_node(new_gcluster);
          ++sendcounts[owner];
        }
        _cluster_weight_deltas[new_gcluster] += w_u;
      }
    }
  }

  struct Message {
    Message() = default;
    GlobalNodeID gcluster;
    GlobalNodeWeight weight;
  };

  std::vector<int> sdispls(comm.size() + 1);
  std::exclusive_scan(sendcounts.begin(), sendcounts.end(), sdispls.begin() + 1,
                      0);
  std::vector<Message> sendbuf(sdispls.back() + sendcounts.back());
  for (auto const& [gcluster, delta] : _cluster_weight_deltas) {
    const PEID pe = _graph->find_owner_of_global_node(gcluster);
    sendbuf[sdispls[pe + 1]++] = {gcluster, delta};
  }

  auto [recvbuf, recvcounts, rdispls] = comm.alltoallv_dispatch(
      send_buf(sendbuf), send_counts(sendcounts), send_displs(sdispls),
      recv_counts_out(), recv_displs_out());

  for (auto const& [gcluster, delta] : recvbuf) {
    change_cluster_weight(gcluster, delta, false);
  }

  // Step 2: reply with the new global cluster weights

  for (auto& [gcluster, delta] : recvbuf) {
    delta = cluster_weight(gcluster);
  }

  auto replies = comm.alltoallv_dispatch(
      send_buf(recvbuf), send_counts(recvcounts), send_displs(rdispls));

  // Step 3: Update cluster weight and revert moves if necessary

  bool violates_max_cluster_weight = false;
  for (auto [gcluster, new_w_gcluster] : replies) {
    GlobalNodeID const old_w_gcluster = cluster_weight(gcluster);

    if (new_w_gcluster > _max_cluster_weight) {
      violates_max_cluster_weight = true;

      GlobalNodeWeight increase_by_others = new_w_gcluster - old_w_gcluster;
      GlobalNodeWeight const increase_by_me = _cluster_weight_deltas[gcluster];

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

void KampingDispatchLP::synchronize_ghost_node_clusters(NodeID const from,
                                                   NodeID const to) {
  Communicator comm(_graph->communicator());

  struct Message {
    Message(GlobalNodeID gnode, GlobalNodeID gcluster)
        : gnode(gnode), gcluster(gcluster) {}
    Message() = default;
    GlobalNodeID gnode;
    GlobalNodeID gcluster;
  };

  std::vector<std::pair<PEID, std::vector<Message>>> sendbuf(comm.size());
  for (PEID pe = 0; pe < comm.size_signed(); ++pe) {
    sendbuf[pe].first = pe;
  }

  Marker created_message_for_pe(comm.size());
  for (NodeID u = from; u < to; ++u) {
    if (_prev_clustering[u] != _clustering[u]) {
      for (auto const& [e, v] : _graph->neighbors(u)) {
        if (_graph->is_owned_node(v)) {
          continue;
        }

        PEID const pe = _graph->ghost_owner(v);
        if (!created_message_for_pe.get(pe)) {
          sendbuf[pe].second.emplace_back(_graph->local_to_global_node(u),
                                          _clustering[u]);
          created_message_for_pe.set(pe);
        }
      }

      created_message_for_pe.reset();
    }
  }

  auto recvbuf =
      with_flattened(sendbuf, comm.size()).call([&](auto... flattened) {
        return comm.alltoallv_dispatch(std::move(flattened)...);
      });

  for (auto const& [gnode, new_gcluster] : recvbuf) {
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

