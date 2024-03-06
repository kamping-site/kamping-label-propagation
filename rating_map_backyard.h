#pragma once

#include <limits>
#include <sparsehash/dense_hash_map>

#include <kaminpar-dist/dkaminpar.h>

namespace kaminpar::dist {
struct RatingMapBackyard {
    RatingMapBackyard() {
        map.set_empty_key(kInvalidGlobalNodeID);
    }

    EdgeWeight& operator[](GlobalNodeID const key) {
        return map[key];
    }

    [[nodiscard]] auto& entries() {
        return map;
    }

    void clear() {
        map.clear();
    }

    [[nodiscard]] std::size_t capacity() const {
        return std::numeric_limits<std::size_t>::max();
    }

    void resize(std::size_t) {}

    google::dense_hash_map<GlobalNodeID, EdgeWeight> map{};
};
} // namespace kaminpar::dist
