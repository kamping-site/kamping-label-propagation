#include "kamping_sparse_label_propagation.h"

#include <kamping/communicator.hpp>
#include <kamping/plugin/alltoall_sparse.hpp>
#include <kamping/types/utility.hpp>

namespace kaminpar::dist {
using namespace ::kamping;
using namespace ::kamping::plugin::sparse_alltoall;

using Communicator = Communicator<std::vector, kamping::plugin::SparseAlltoall>;

using MsgBuf = std::vector<
    std::pair<PEID, std::vector<std::pair<std::int64_t, std::int64_t>>>>;

KampingSparseLP::KampingSparseLP(Context const& ctx) : BaseLP(ctx) {}

void KampingSparseLP::enforce_max_cluster_weights(NodeID const from,
                                                  NodeID const to) {
  Communicator comm(_graph->communicator());
  const PEID size = comm.size();

  // Step 1: Aggregate the weight deltas

  _cluster_weight_deltas.clear();
  for (NodeID const u : _graph->nodes(from, to)) {
    GlobalNodeID const old_gcluster = _prev_clustering[u];
    GlobalNodeID const new_gcluster = _clustering[u];

    if (old_gcluster != new_gcluster) {
      NodeWeight const w_u = _graph->node_weight(u);

      if (!_graph->is_owned_global_node(old_gcluster)) {
        _cluster_weight_deltas[old_gcluster] -= w_u;
      }

      if (!_graph->is_owned_global_node(new_gcluster)) {
        _cluster_weight_deltas[new_gcluster] += w_u;
      }
    }
  }

  MsgBuf sendbuf(size);
  MsgBuf recvbuf(size);
  for (PEID pe = 0; pe < size; ++pe) {
    sendbuf[pe].first = pe;
    recvbuf[pe].first = pe;
  }

  for (auto const& [gcluster, delta] : _cluster_weight_deltas) {
    const PEID pe = _graph->find_owner_of_global_node(gcluster);
    sendbuf[pe].second.push_back({gcluster, delta});
  }

  comm.alltoallv_sparse(sparse_send_buf(sendbuf),
                        on_message([&](auto const& msg) {
                          auto msgs = msg.recv();
                          for (auto const& [gcluster, delta] : msgs) {
                            change_cluster_weight(gcluster, delta, false);
                          }
                          recvbuf[msg.source()].second = std::move(msgs);
                        }));

  // Step 2: reply with the new global cluster weights

  for (PEID pe = 0; pe < size; ++pe) {
    for (auto& [gcluster, delta] : recvbuf[pe].second) {
      delta = cluster_weight(gcluster);
    }
  }

  bool violates_max_cluster_weight = false;

  comm.alltoallv_sparse(
      sparse_send_buf(recvbuf), on_message([&](auto const& msg) {
        auto msgs = msg.recv();
        for (auto [gcluster, new_w_gcluster] : msgs) {
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

          change_cluster_weight(gcluster, -old_w_gcluster + new_w_gcluster,
                                true);
        }
      }));

  // Step 3: Revert moves if necessary

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

void KampingSparseLP::synchronize_ghost_node_clusters(NodeID const from,
                                                      NodeID const to) {
  Communicator comm(_graph->communicator());
  const PEID size = comm.size();

  MsgBuf sendbuf(size);
  for (PEID pe = 0; pe < size; ++pe) {
    sendbuf[pe].first = pe;
  }

  Marker created_message_for_pe(size);
  for (NodeID u = from; u < to; ++u) {
    if (_prev_clustering[u] != _clustering[u]) {
      for (auto const& [e, v] : _graph->neighbors(u)) {
        if (_graph->is_owned_node(v)) {
          continue;
        }

        PEID const pe = _graph->ghost_owner(v);
        if (!created_message_for_pe.get(pe)) {
          sendbuf[pe].second.push_back(
              {_graph->local_to_global_node(u), _clustering[u]});
          created_message_for_pe.set(pe);
        }
      }

      created_message_for_pe.reset();
    }
  }

  comm.alltoallv_sparse(
      sparse_send_buf(sendbuf), on_message([&](auto const& msg) {
        auto const msgs = msg.recv();
        for (auto const& [gnode, new_gcluster] : msgs) {
          NodeID const lnode = _graph->global_to_local_node(gnode);

          _clustering[lnode] = new_gcluster;

          // Update the cluster weights if we do not already have the exact
          // cluster weight due to enforce_max_cluster_weights(): (i) if we own
          // the clusters, we have its exact weight (ii) if we have the cluster
          // in our _cluster_weight_deltas, we have its exact weight In all
          // other cases, we do not know its exact weight: track the delta
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
      }));
}
}  // namespace kaminpar::dist

