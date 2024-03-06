#pragma once

#include "base_label_propagation.h"

namespace kaminpar::dist {
class KaminparLP : public BaseLP {
 public:
  explicit KaminparLP(Context const& ctx);

 protected:
  void enforce_max_cluster_weights(NodeID from, NodeID to);
  void synchronize_ghost_node_clusters(NodeID from, NodeID to);
};
}  // namespace kaminpar::dist
