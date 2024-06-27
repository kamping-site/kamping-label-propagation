// clang-format off
#include <kaminpar-cli/dkaminpar_arguments.h>
#include <kaminpar-dist/dkaminpar.h>
// clang-format on

#include <kagen.h>
#include <mpi.h>

#include "kaminpar_label_propagation.h"
#include "kamping_dispatch_label_propagation.h"
#include "kamping_label_propagation.h"
#include "kamping_sparse_label_propagation.h"
#include "mpi_label_propagation.h"

using namespace kaminpar;
using namespace kaminpar::dist;

namespace {
std::unique_ptr<GlobalClusterer> create_kaminpar_clusterer(Context const& ctx) {
  return std::make_unique<KaminparLP>(ctx);
}

std::unique_ptr<GlobalClusterer> create_kamping_dispatch_clusterer(
    Context const& ctx) {
  return std::make_unique<KampingDispatchLP>(ctx);
}

std::unique_ptr<GlobalClusterer> create_kamping_sparse_clusterer(
    Context const& ctx) {
  return std::make_unique<KampingSparseLP>(ctx);
}

std::unique_ptr<GlobalClusterer> create_kamping_clusterer(Context const& ctx) {
  return std::make_unique<KampingLP>(ctx);
}

std::unique_ptr<GlobalClusterer> create_mpi_clusterer(Context const& ctx) {
  return std::make_unique<MpiLP>(ctx);
}

template <typename Clusterer>
Context create_context(Clusterer& clusterer) {
  Context ctx = create_default_context();
  ctx.coarsening.global_clustering_algorithm =
      GlobalClusteringAlgorithm::EXTERNAL;
  ctx.coarsening.external_global_clustering_algorithm = clusterer;
  return ctx;
}

struct ApplicationContext {
  int seed = 0;

  BlockID k = 2;

  kagen::FileFormat io_format = kagen::FileFormat::EXTENSION;
  kagen::GraphDistribution io_distribution =
      kagen::GraphDistribution::BALANCE_EDGES;

  std::string graph_filename = "";
};

NodeID load_kagen_graph(ApplicationContext const& app, dKaMinPar& partitioner) {
  using namespace kagen;

  KaGen generator(MPI_COMM_WORLD);
  generator.UseCSRRepresentation();
  Graph graph = [&] {
    if (std::find(app.graph_filename.begin(), app.graph_filename.end(), ';') !=
        app.graph_filename.end()) {
      return generator.GenerateFromOptionString(app.graph_filename);
    } else {
      return generator.ReadFromFile(app.graph_filename, app.io_format,
                                    app.io_distribution);
    }
  }();

  std::vector<GlobalNodeID> vtxdist = BuildVertexDistribution<unsigned long>(
      graph, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
  std::vector<GlobalEdgeID> xadj = graph.TakeXadj<GlobalEdgeID>();
  std::vector<GlobalNodeID> adjncy = graph.TakeAdjncy<GlobalNodeID>();
  std::vector<GlobalNodeWeight> vwgt =
      graph.TakeVertexWeights<GlobalNodeWeight>();
  std::vector<GlobalEdgeWeight> adjwgt =
      graph.TakeEdgeWeights<GlobalEdgeWeight>();

  bool no_vwgt = vwgt.empty(), no_adjwgt = adjwgt.empty();
  MPI_Allreduce(MPI_IN_PLACE, &no_vwgt, 1, MPI_CXX_BOOL, MPI_LAND,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &no_adjwgt, 1, MPI_CXX_BOOL, MPI_LAND,
                MPI_COMM_WORLD);

  partitioner.import_graph(vtxdist.data(), xadj.data(), adjncy.data(),
                           no_vwgt ? nullptr : vwgt.data(),
                           no_adjwgt ? nullptr : adjwgt.data());

  return graph.vertex_range.second - graph.vertex_range.first;
}

void perform_warmup_alltoallv() {
  PEID size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<int> sendbuf(size, 42);
  std::vector<int> sendcounts(size, 1);
  std::vector<int> sdispls(size, 0);
  std::iota(sdispls.begin(), sdispls.end(), 0);
  std::vector<int> recvbuf(size, 0);
  std::vector<int> recvcounts(size, 1);
  std::vector<int> rdispls(size, 0);
  std::iota(rdispls.begin(), rdispls.end(), 0);
  MPI_Alltoallv(sendbuf.data(), sendcounts.data(), sdispls.data(), MPI_INT,
                recvbuf.data(), recvcounts.data(), rdispls.data(), MPI_INT,
                MPI_COMM_WORLD);
}
}  // namespace

int main(int argc, char* argv[]) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

  CLI::App cli;
  ApplicationContext app;
  Context ctx = create_default_context();

  cli.add_option_function<std::string>(
         "-P,--preset",
         [&](std::string const preset) {
           if (preset == "mpi") {
             ctx = create_context(create_mpi_clusterer);
           } else if (preset == "kamping") {
             ctx = create_context(create_kamping_clusterer);
           } else if (preset == "kamping-sparse") {
             ctx = create_context(create_kamping_sparse_clusterer);
           } else if (preset == "kamping-dispatch") {
             ctx = create_context(create_kamping_dispatch_clusterer);
           } else if (preset == "kaminpar") {
             ctx = create_context(create_kaminpar_clusterer);
           } else {
             ctx = create_context_by_preset_name(preset);
           }
         })
      ->check(CLI::IsMember({"inorder-default", "default", "mpi",
                             "kamping-dispatch", "kamping", "kamping-sparse",
                             "kaminpar"}))
      ->required();
  cli.add_option("-G,--graph,--kagen_option_string", app.graph_filename)->required();

  cli.add_option("-k,--k", app.k, "Number of blocks in the partition.")
      ->capture_default_str();
  cli.add_option("-s,--seed", app.seed, "Seed for random number generation.")
      ->capture_default_str();

  CLI11_PARSE(cli, argc, argv);

  perform_warmup_alltoallv();

  dKaMinPar partitioner(MPI_COMM_WORLD, 1, ctx);
  dKaMinPar::reseed(app.seed);

  partitioner.set_output_level(OutputLevel::EXPERIMENT);
  partitioner.context().debug.graph_filename = app.graph_filename;

  NodeID const n = load_kagen_graph(app, partitioner);

  std::vector<BlockID> partition(n);
  partitioner.compute_partition(app.k, partition.data());

  return MPI_Finalize();
}
