#pragma once

#include "base_label_propagation.h"

namespace kaminpar::dist {
class MpiLP : public BaseLP {
 public:
  explicit MpiLP(Context const& ctx);

 protected:
  void enforce_max_cluster_weights(NodeID from, NodeID to) final;
  void synchronize_ghost_node_clusters(NodeID from, NodeID to) final;
};
}  // namespace kaminpar::dist
