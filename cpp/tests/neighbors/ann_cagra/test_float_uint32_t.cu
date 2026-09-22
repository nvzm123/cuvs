/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_cagra.cuh"

#include <utility>
#include <vector>

namespace cuvs::neighbors::cagra {

typedef AnnCagraTest<float, float, std::uint32_t> AnnCagraTestF_U32;
TEST_P(AnnCagraTestF_U32, AnnCagra_U32) { this->testCagra<uint32_t>(); }
TEST_P(AnnCagraTestF_U32, AnnCagra_I64) { this->testCagra<int64_t>(); }

typedef AnnCagraAddNodesTest<float, float, std::uint32_t> AnnCagraAddNodesTestF_U32;
TEST_P(AnnCagraAddNodesTestF_U32, AnnCagraAddNodes) { this->testCagra(); }

typedef AnnCagraFilterTest<float, float, std::uint32_t> AnnCagraFilterTestF_U32;
TEST_P(AnnCagraFilterTestF_U32, AnnCagra) { this->testCagra(); }

typedef AnnCagraIndexMergeTest<float, float, std::uint32_t> AnnCagraIndexMergeTestF_U32;
TEST_P(AnnCagraIndexMergeTestF_U32, AnnCagraIndexMerge_U32) { this->testCagra<uint32_t>(); }
TEST_P(AnnCagraIndexMergeTestF_U32, AnnCagraIndexMerge_I64) { this->testCagra<int64_t>(); }

typedef AnnCagraIndexFilteredMergeTest<float, float, std::uint32_t>
  AnnCagraIndexFilteredMergeTestF_U32;
TEST_P(AnnCagraIndexFilteredMergeTestF_U32, AnnCagraIndexFilteredMerge_U32)
{
  this->testCagra<uint32_t>();
}

INSTANTIATE_TEST_CASE_P(AnnCagraTest, AnnCagraTestF_U32, ::testing::ValuesIn(inputs));
INSTANTIATE_TEST_CASE_P(AnnCagraAddNodesTest,
                        AnnCagraAddNodesTestF_U32,
                        ::testing::ValuesIn(inputs_addnode));
INSTANTIATE_TEST_CASE_P(AnnCagraFilterTest,
                        AnnCagraFilterTestF_U32,
                        ::testing::ValuesIn(inputs_filtering));
INSTANTIATE_TEST_CASE_P(AnnCagraIndexMergeTest,
                        AnnCagraIndexMergeTestF_U32,
                        ::testing::ValuesIn(inputs));

INSTANTIATE_TEST_CASE_P(AnnCagraIndexFilteredMergeTest,
                        AnnCagraIndexFilteredMergeTestF_U32,
                        ::testing::ValuesIn(inputs));

typedef AnnCagraMultiPartitionTest<float, float, std::uint32_t> AnnCagraMultiPartitionTestF_U32;
TEST_P(AnnCagraMultiPartitionTestF_U32, Search) { this->testSearch(); }
TEST_P(AnnCagraMultiPartitionTestF_U32, FilteredSearch) { this->testFilteredSearch(); }

INSTANTIATE_TEST_CASE_P(AnnCagraMultiPartitionTest,
                        AnnCagraMultiPartitionTestF_U32,
                        ::testing::ValuesIn(inputs_mp));

// Builds one CAGRA index per {metric, graph_degree} spec over a shared random dataset, then asserts
// a multi-partition search over them throws. Shared by the rejection tests below, which each
// violate one "all partitions must be uniform / supported" precondition. The rejections are
// invariant to dtype / layout, so they are checked once here instead of swept across the fixture.
namespace {
void expect_multi_partition_search_throws(
  const std::vector<std::pair<cuvs::distance::DistanceType, int>>& partition_specs,
  const cagra::search_params& search_params)
{
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  constexpr int n_rows = 256, dim = 8, n_queries = 10, k = 4;
  const int num_partitions = static_cast<int>(partition_specs.size());
  const int part_size      = n_rows / num_partitions;

  rmm::device_uvector<float> database(static_cast<size_t>(n_rows) * dim, stream);
  rmm::device_uvector<float> queries(static_cast<size_t>(n_queries) * dim, stream);
  raft::random::RngState r(1234ULL);
  InitDataset(handle, database.data(), n_rows, dim, cuvs::distance::DistanceType::L2Expanded, r);
  InitDataset(handle, queries.data(), n_queries, dim, cuvs::distance::DistanceType::L2Expanded, r);
  raft::resource::sync_stream(handle);

  std::vector<cagra::index<float, std::uint32_t>> part_indices;
  // An index only holds a view, so any padded copy has to outlive it.
  std::vector<cuvs::neighbors::test::padded_device_matrix_for_cagra<float>> part_padded;
  part_padded.reserve(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    const auto [metric, graph_degree] = partition_specs[i];
    cagra::index_params index_params;
    index_params.metric                    = metric;
    index_params.graph_degree              = graph_degree;
    index_params.intermediate_graph_degree = graph_degree * 2;
    index_params.graph_build_params =
      graph_build_params::nn_descent_params(index_params.intermediate_graph_degree, metric);
    auto view = raft::make_device_matrix_view<const float, int64_t>(
      database.data() + static_cast<size_t>(i) * part_size * dim, part_size, dim);
    part_padded.emplace_back(handle, view);
    auto const& padded = part_padded.back().view;
    part_indices.push_back(cagra::build(handle, index_params, padded));
    auto& part_index = part_indices.back();
    part_index       = cagra::update_dataset(handle, std::move(part_index), padded);
  }
  std::vector<const cagra::index<float, std::uint32_t>*> index_ptrs;
  for (auto& idx : part_indices) {
    index_ptrs.push_back(&idx);
  }

  const size_t out_size = static_cast<size_t>(n_queries) * k;
  rmm::device_uvector<uint32_t> partition_ids(out_size, stream);
  rmm::device_uvector<uint32_t> neighbors(out_size, stream);
  rmm::device_uvector<float> distances(out_size, stream);

  auto queries_view =
    raft::make_device_matrix_view<const float, int64_t>(queries.data(), n_queries, dim);
  auto part_ids_view =
    raft::make_device_matrix_view<uint32_t, int64_t>(partition_ids.data(), n_queries, k);
  auto neighbors_view =
    raft::make_device_matrix_view<uint32_t, int64_t>(neighbors.data(), n_queries, k);
  auto dists_view = raft::make_device_matrix_view<float, int64_t>(distances.data(), n_queries, k);

  EXPECT_THROW(
    cagra::search(
      handle, search_params, index_ptrs, queries_view, part_ids_view, neighbors_view, dists_view),
    std::exception);
}
}  // namespace

// MULTI_KERNEL is intentionally unsupported in the multi-partition path; the call must fail rather
// than silently fall back.
TEST(AnnCagraMultiPartition, MultiKernelRejected)
{
  cagra::search_params search_params;
  search_params.algo = search_algo::MULTI_KERNEL;
  expect_multi_partition_search_throws({{cuvs::distance::DistanceType::L2Expanded, 16},
                                        {cuvs::distance::DistanceType::L2Expanded, 16}},
                                       search_params);
}

// The shared plan descriptor and the cross-partition select_k direction are derived from
// indices[0], so all partitions must share one metric; a mismatch must be rejected.
TEST(AnnCagraMultiPartition, MixedMetricRejected)
{
  expect_multi_partition_search_throws({{cuvs::distance::DistanceType::L2Expanded, 16},
                                        {cuvs::distance::DistanceType::InnerProduct, 16}},
                                       cagra::search_params{});
}

// The shared plan descriptor is sized from indices[0]'s graph degree, so all partitions must share
// one graph degree; a mismatch must be rejected.
TEST(AnnCagraMultiPartition, MixedGraphDegreeRejected)
{
  expect_multi_partition_search_throws({{cuvs::distance::DistanceType::L2Expanded, 16},
                                        {cuvs::distance::DistanceType::L2Expanded, 32}},
                                       cagra::search_params{});
}

// A manually assembled graph may not know its expected dataset dimension. Attaching its first
// dataset remains supported and makes that dimension observable.
TEST(AnnCagraIndexMetadata, UnknownDimensionAcceptsFirstDataset)
{
  raft::resources res;
  auto graph = raft::make_host_matrix<uint32_t, int64_t>(4, 1);
  for (int64_t row = 0; row < graph.extent(0); ++row) {
    graph(row, 0) = static_cast<uint32_t>((row + 1) % graph.extent(0));
  }

  cagra::device_padded_index<float> graph_only(res);
  graph_only.update_graph(res, raft::make_const_mdspan(graph.view()));
  EXPECT_EQ(graph_only.dim(), 0);

  auto queries   = raft::make_device_matrix<float, int64_t>(res, 1, 3);
  auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, 1, 1);
  auto distances = raft::make_device_matrix<float, int64_t>(res, 1, 1);
  try {
    cagra::search(res,
                  cagra::search_params{},
                  graph_only,
                  raft::make_const_mdspan(queries.view()),
                  neighbors.view(),
                  distances.view());
    FAIL() << "search accepted a graph without an attached dataset";
  } catch (std::exception const& error) {
    EXPECT_NE(std::string(error.what()).find("without an attached dataset"), std::string::npos)
      << error.what();
  }

  auto storage = raft::make_device_matrix<float, int64_t>(res, 4, 4);
  cuvs::neighbors::device_padded_dataset_view<float, int64_t> dataset(
    raft::make_const_mdspan(storage.view()), 3);
  auto attached = cagra::update_dataset(res, std::move(graph_only), dataset);
  EXPECT_EQ(attached.dim(), 3);
  EXPECT_EQ(attached.dataset().n_rows(), 4);
}

TEST(AnnCagraIndexMetadata, ExplicitDimensionIsAvailableWithoutDataset)
{
  raft::resources res;
  cagra::device_padded_index<float> legacy_default(res);
  cagra::device_padded_index<float> legacy_metric(res, cuvs::distance::DistanceType::InnerProduct);
  cagra::device_padded_index<float> graph_only(res, cuvs::distance::DistanceType::L2Expanded, 3);

  EXPECT_EQ(legacy_default.metric(), cuvs::distance::DistanceType::L2Expanded);
  EXPECT_EQ(legacy_default.dim(), 0);
  EXPECT_EQ(legacy_metric.metric(), cuvs::distance::DistanceType::InnerProduct);
  EXPECT_EQ(legacy_metric.dim(), 0);
  EXPECT_EQ(graph_only.dataset().n_rows(), 0);
  EXPECT_EQ(graph_only.dim(), 3);
}

}  // namespace cuvs::neighbors::cagra
