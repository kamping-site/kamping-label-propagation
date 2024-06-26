cmake_minimum_required(VERSION 3.20)
project(KaMPIngLabelPropagation CXX)

add_subdirectory(external/kamping)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(KAGEN_NODEPS ON)
set(KAGEN_BUILD_APPS OFF)
set(KAGEN_BUILD_EXAMPLES OFF)
set(KAGEN_BUILD_TESTS OFF)
set(KAGEN_USE_SPARSEHASH OFF)
add_subdirectory(external/KaGen)

set(KAMINPAR_BUILD_DISTRIBUTED ON)
set(KAMINPAR_BUILD_TESTS OFF)
set(KAMINPAR_BUILD_APPS OFF)
set(KAMINPAR_BUILD_BENCHMARKS OFF)
set(KAMINPAR_ENABLE_STATISTICS OFF)
add_subdirectory(external/KaMinPar)

set(SOURCE_FILES
    main.cc
    rating_map_backyard.h
    base_label_propagation.h
    base_label_propagation.cc
    mpi_label_propagation.h
    mpi_label_propagation.cc
    kamping_label_propagation.h
    kamping_label_propagation.cc
    kamping_sparse_label_propagation.h
    kamping_sparse_label_propagation.cc
    kamping_dispatch_label_propagation.h
    kamping_dispatch_label_propagation.cc
    kaminpar_label_propagation.cc
    kaminpar_label_propagation.h)

add_executable(KaMPIngLabelPropagation ${SOURCE_FILES})
target_include_directories(KaMPIngLabelPropagation PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(KaMPIngLabelPropagation PRIVATE 
    KaGen::KaGen
    KaMinPar::dKaMinPar 
    KaMinPar::dKaMinParCLI 
    kamping::kamping)
target_compile_options(KaMPIngLabelPropagation PRIVATE -mcx16 -g)
