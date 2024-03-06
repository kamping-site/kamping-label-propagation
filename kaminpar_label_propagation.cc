#include "kaminpar_label_propagation.h"

#include <kaminpar-dist/graphutils/communication.h>
#include <kaminpar-mpi/wrapper.h>

namespace kaminpar::dist {
KaminparLP::KaminparLP(Context const& ctx) : BaseLP(ctx) {}

void KaminparLP::enforce_max_cluster_weights(NodeID const from,
                                             NodeID const to) {
  const PEID size = mpi::get_comm_size(_graph->communicator());

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

  struct Message {
    GlobalNodeID gcluster;
    GlobalNodeWeight weight;
  };

  std::vector<std::vector<Message>> sendbufs(size);
  for (auto const& [gcluster, delta] : _cluster_weight_deltas) {
    const PEID pe = _graph->find_owner_of_global_node(gcluster);
    sendbufs[pe].push_back({gcluster, delta});
  }

  auto recvbufs =
      mpi::sparse_alltoall_get<Message>(sendbufs, _graph->communicator());

  for (PEID pe = 0; pe < size; ++pe) {
    for (auto const& [gcluster, delta] : recvbufs[pe]) {
      change_cluster_weight(gcluster, delta, false);
    }
  }

  // Step 2: reply with the new global cluster weights

  for (PEID pe = 0; pe < size; ++pe) {
    for (auto& [gcluster, delta] : recvbufs[pe]) {
      delta = cluster_weight(gcluster);
    }
  }

  auto replies =
      mpi::sparse_alltoall_get<Message>(recvbufs, _graph->communicator());

  // Step 3: Update cluster weight and revert moves if necessary

  bool violates_max_cluster_weight = false;
  for (PEID pe = 0; pe < size; ++pe) {
    for (auto [gcluster, new_w_gcluster] : replies[pe]) {
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

void KaminparLP::synchronize_ghost_node_clusters(NodeID const from,
                                                 NodeID const to) {
  struct Message {
    GlobalNodeID gnode;
    GlobalNodeID gcluster;
  };

  mpi::graph::sparse_alltoall_interface_to_pe<Message>(
      *_graph, from, to,
      [&](NodeID const lnode) {
        return _prev_clustering[lnode] != _clustering[lnode];
      },
      [&](NodeID const lnode) -> Message {
        return {_graph->local_to_global_node(lnode), _clustering[lnode]};
      },
      [&](auto const& buffer, const PEID owner) {
        for (auto const& [gnode, new_gcluster] : buffer) {
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
      });
}
}  // namespace kaminpar::dist

