#pragma once

#include <kaminpar-common/datastructures/dynamic_map.h>
#include <kaminpar-common/datastructures/rating_map.h>
#include <kaminpar-common/datastructures/static_array.h>
#include <kaminpar-dist/coarsening/clustering/clusterer.h>
#include <kaminpar-dist/datastructures/distributed_graph.h>
#include <kaminpar-dist/dkaminpar.h>

#include <sparsehash/dense_hash_map>

#include "rating_map_backyard.h"

namespace kaminpar::dist {
class BaseLP : public Clusterer<GlobalNodeID> {
 public:
  explicit BaseLP(Context const& ctx);

  void initialize(DistributedGraph const& graph);

  ClusterArray& cluster(DistributedGraph const& graph,
                        GlobalNodeWeight max_cluster_weight) final;

 protected:
  GlobalNodeID process_chunk(NodeID from, NodeID to);
  bool handle_node(NodeID node);
  GlobalNodeID find_best_cluster(NodeID u, NodeWeight w_u, GlobalNodeID C_u);
  GlobalNodeWeight cluster_weight(GlobalNodeID C);

  void init_cluster_weight(GlobalNodeID lcluster,
                           GlobalNodeWeight const weight);
  void move_node(NodeID lnode, NodeWeight w_lnode, GlobalNodeID old_gcluster,
                 GlobalNodeID new_gcluster);
  void change_cluster_weight(GlobalNodeID gcluster,
                             GlobalNodeWeight const delta, bool must_exist);

  virtual void enforce_max_cluster_weights(NodeID from, NodeID to) = 0;
  virtual void synchronize_ghost_node_clusters(NodeID from, NodeID to) = 0;

  void cluster_isolated_nodes(NodeID from = 0,
                              NodeID to = std::numeric_limits<NodeID>::max());

  google::dense_hash_map<GlobalNodeID, GlobalNodeWeight>
      _global_cluster_weights;
  google::dense_hash_map<GlobalNodeID, GlobalNodeWeight> _cluster_weight_deltas;

  RatingMap<EdgeWeight, GlobalNodeID, RatingMapBackyard> _ratings{0};

  // Initialized by initialize()
  DistributedGraph const* _graph = nullptr;

  StaticArray<GlobalNodeWeight> _local_cluster_weights{};
  ClusterArray _clustering{};
  ClusterArray _prev_clustering{};

  // Initialized by cluster()
  GlobalNodeWeight _max_cluster_weight;

  // Initialized by c'tor
  Context const& _ctx;
  LabelPropagationCoarseningContext const& _lp_ctx;
};
}  // namespace kaminpar::dist
