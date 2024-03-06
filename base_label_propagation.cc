#include "base_label_propagation.h"

#include <kaminpar-common/random.h>
#include <kaminpar-dist/timer.h>
#include <mpi.h>

namespace kaminpar::dist {
BaseLP::BaseLP(Context const& ctx)
    : _ctx(ctx), _lp_ctx(ctx.coarsening.global_lp) {
  _global_cluster_weights.set_empty_key(kInvalidGlobalNodeID);
  _cluster_weight_deltas.set_empty_key(kInvalidGlobalNodeID);
}

void BaseLP::initialize(DistributedGraph const& graph) {
  _graph = &graph;

  MPI_Barrier(graph.communicator());
  SCOPED_TIMER("Label propagation clustering");

  _global_cluster_weights.clear();
  _cluster_weight_deltas.clear();
  _prev_clustering.resize(_graph->total_n());
  _clustering.resize(_graph->total_n());
  _local_cluster_weights.resize(_graph->n());
  for (NodeID lu : _graph->all_nodes()) {
    _prev_clustering[lu] = _graph->local_to_global_node(lu);
    _clustering[lu] = _graph->local_to_global_node(lu);
  }
  for (NodeID lu : _graph->nodes()) {
    _local_cluster_weights[lu] = _graph->node_weight(lu);
  }
  for (NodeID lu : _graph->ghost_nodes()) {
    _global_cluster_weights[_graph->local_to_global_node(lu)] =
        _graph->node_weight(lu);
  }

  _ratings.change_max_size(_graph->total_n());
}

GlobalClusterer::ClusterArray& BaseLP::cluster(
    DistributedGraph const& graph, GlobalNodeWeight max_cluster_weight) {
  MPI_Barrier(graph.communicator());
  SCOPED_TIMER("Label propagation clustering");

  _max_cluster_weight = max_cluster_weight;

  int const num_chunks = _lp_ctx.chunks.compute(_ctx.parallel);
  GlobalNodeID num_moved_nodes_overall = 0;

  for (int it = 0; it < _lp_ctx.num_iterations; ++it) {
    GlobalNodeID num_moved_nodes = 0;
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
      auto const [from, to] =
          math::compute_local_range<NodeID>(_graph->n(), num_chunks, chunk);
      num_moved_nodes += process_chunk(from, to);
    }

    num_moved_nodes_overall += num_moved_nodes;
    if (num_moved_nodes == 0) {
      break;
    }
  }

  // If we couldn't reduce the number of nodes by at least x2, try to shrink the
  // graph further by clustering isolated nodes
  if (2 * num_moved_nodes_overall < _graph->global_n()) {
    cluster_isolated_nodes();
  }

  return _clustering;
}

GlobalNodeID BaseLP::process_chunk(NodeID const from, NodeID const to) {
  GlobalNodeID num_moved_nodes = 0;

  for (NodeID u = from; u < to; ++u) {
    if (handle_node(u)) {
      ++num_moved_nodes;
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, &num_moved_nodes, 1,
                mpi::type::get<GlobalNodeID>(), MPI_SUM,
                _graph->communicator());

  if (num_moved_nodes > 0) {
    enforce_max_cluster_weights(from, to);
    synchronize_ghost_node_clusters(from, to);
  }

  _graph->pfor_nodes(from, to, [&](NodeID const lnode) {
    _prev_clustering[lnode] = _clustering[lnode];
  });

  return num_moved_nodes;
}

bool BaseLP::handle_node(NodeID const u) {
  NodeWeight const w_u = _graph->node_weight(u);
  GlobalNodeID const old_gcluster = _prev_clustering[u];
  GlobalNodeID const new_gcluster = find_best_cluster(u, w_u, old_gcluster);

  if (old_gcluster != new_gcluster) {
    move_node(u, w_u, old_gcluster, new_gcluster);
  }

  return old_gcluster != new_gcluster;
}

GlobalNodeID BaseLP::find_best_cluster(NodeID const u, NodeWeight const w_u,
                                       GlobalNodeID const old_gcluster) {
  return _ratings.execute(_graph->degree(u), [&](auto& map) {
    for (auto const [e, v] : _graph->neighbors(u)) {
      map[_clustering[v]] += _graph->edge_weight(e);
    }

    Random& rand = Random::instance();
    EdgeWeight max_conn = 0;
    GlobalNodeID new_gcluster = old_gcluster;

    for (auto const [gcluster, conn] : map.entries()) {
      if ((conn > max_conn || (conn == max_conn && rand.random_bool())) &&
          cluster_weight(gcluster) + w_u <= _max_cluster_weight) {
        new_gcluster = gcluster;
        max_conn = conn;
      }
    }

    map.clear();
    return new_gcluster;
  });
}

GlobalNodeWeight BaseLP::cluster_weight(GlobalNodeID const gcluster) {
  if (_graph->is_owned_global_node(gcluster)) {
    return _local_cluster_weights[_graph->global_to_local_node(gcluster)];
  } else {
    return _global_cluster_weights[gcluster];
  }
}

void BaseLP::init_cluster_weight(GlobalNodeID const lcluster,
                                 GlobalNodeWeight const weight) {
  if (_graph->is_owned_node(lcluster)) {
    _local_cluster_weights[lcluster] = weight;
  } else {
    auto const gcluster =
        _graph->local_to_global_node(static_cast<NodeID>(lcluster));
    _global_cluster_weights[gcluster] = weight;
  }
}

void BaseLP::move_node(NodeID const lnode, NodeWeight const w_lnode,
                       GlobalNodeID const old_gcluster,
                       GlobalNodeID const new_gcluster) {
  _clustering[lnode] = new_gcluster;
  change_cluster_weight(old_gcluster, -w_lnode, true);
  change_cluster_weight(new_gcluster, w_lnode, true);
}

void BaseLP::change_cluster_weight(GlobalNodeID const gcluster,
                                   GlobalNodeWeight const delta,
                                   [[maybe_unused]] bool const must_exist) {
  if (_graph->is_owned_global_node(gcluster)) {
    _local_cluster_weights[_graph->global_to_local_node(gcluster)] += delta;
  } else {
    _global_cluster_weights[gcluster] += delta;
  }
}

void BaseLP::cluster_isolated_nodes(NodeID const from, NodeID const to) {
  GlobalNodeID cur_C = kInvalidGlobalNodeID;
  GlobalNodeWeight cur_w_C = kInvalidGlobalNodeWeight;

  for (NodeID const u :
       _graph->nodes(from, std::min<NodeID>(to, _graph->n()))) {
    if (_graph->degree(u) == 0) {
      auto const C_u = _prev_clustering[u];
      auto const w_C_u = cluster_weight(C_u);

      if (cur_C != kInvalidGlobalNodeID &&
          cur_w_C + w_C_u <= _max_cluster_weight) {
        change_cluster_weight(cur_C, w_C_u, true);
        _clustering[u] = cur_C;
        cur_w_C += w_C_u;
      } else {
        cur_C = C_u;
        cur_w_C = w_C_u;
      }
    }
  }
}
}  // namespace kaminpar::dist

