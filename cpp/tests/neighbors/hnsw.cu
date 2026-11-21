/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "ann_utils.cuh"
#include "cagra_padded_build_helpers.cuh"

#include <cstdint>
#include <cuda/stream>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/hnsw.hpp>
#include <hnswlib/hnswlib.h>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/pinned_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/itertools.hpp>

#include "naive_knn.cuh"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cuvs::neighbors::hnsw {

struct AnnHNSWInputs {
  int n_rows;
  int dim;
  int graph_degree;
  cuvs::distance::DistanceType metric;
  int k;
  int n_queries;
  int ef;
  double min_recall;
};

inline ::std::ostream& operator<<(::std::ostream& os, const AnnHNSWInputs& p)
{
  os << "dataset shape=" << p.n_rows << "x" << p.dim << ", graph_degree=" << p.graph_degree
     << ", metric="
     << cuvs::neighbors::print_metric{static_cast<cuvs::distance::DistanceType>((int)p.metric)}
     << ", ef=" << (p.ef);
  return os;
}

template <typename DistanceT, typename DataT, typename IdxT>
class AnnHNSWTest : public ::testing::TestWithParam<AnnHNSWInputs> {
 public:
  AnnHNSWTest()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<AnnHNSWInputs>::GetParam()),
      database(0, stream_),
      queries(0, stream_)
  {
  }

 protected:
  void testHNSW()
  {
    std::vector<IdxT> indices_HNSW(ps.n_queries * ps.k);
    std::vector<DistanceT> distances_HNSW(ps.n_queries * ps.k);
    std::vector<IdxT> indices_naive(ps.n_queries * ps.k);
    std::vector<DistanceT> distances_naive(ps.n_queries * ps.k);

    std::vector<DataT> queries_h(ps.n_queries * ps.dim);
    raft::update_host(queries_h.data(), queries.data(), ps.n_queries * ps.dim, stream_);

    {
      rmm::device_uvector<DistanceT> distances_naive_dev(ps.n_queries * ps.k, stream_);
      rmm::device_uvector<IdxT> indices_naive_dev(ps.n_queries * ps.k, stream_);
      naive_knn<DistanceT, DataT, IdxT>(handle_,
                                        distances_naive_dev.data(),
                                        indices_naive_dev.data(),
                                        queries.data(),
                                        database.data(),
                                        ps.n_queries,
                                        ps.n_rows,
                                        ps.dim,
                                        ps.k,
                                        ps.metric);
      raft::update_host(
        indices_naive.data(), indices_naive_dev.data(), ps.n_queries * ps.k, stream_);
      raft::update_host(
        distances_naive.data(), distances_naive_dev.data(), ps.n_queries * ps.k, stream_);
      raft::resource::sync_stream(handle_);
    }

    {
      cuvs::neighbors::cagra::index_params index_params;
      index_params.metric                    = ps.metric;
      index_params.graph_degree              = ps.graph_degree;
      index_params.intermediate_graph_degree = 2 * ps.graph_degree;

      auto database_view = raft::make_device_matrix_view<const DataT, int64_t>(
        (const DataT*)database.data(), ps.n_rows, ps.dim);
      cuvs::neighbors::test::padded_device_matrix_for_cagra<DataT> padded(handle_, database_view);

      auto index = cuvs::neighbors::cagra::build(handle_, index_params, padded.view);
      index      = cuvs::neighbors::cagra::update_dataset(handle_, std::move(index), padded.view);
      raft::resource::sync_stream(handle_);

      cuvs::neighbors::hnsw::search_params search_params;
      search_params.ef = ps.ef;
      cuvs::neighbors::hnsw::index_params hnsw_params;
      auto hnsw_index = cuvs::neighbors::hnsw::from_cagra(handle_, hnsw_params, index);
      auto queries_HNSW_view =
        raft::make_host_matrix_view<DataT, int64_t>(queries_h.data(), ps.n_queries, ps.dim);
      auto indices_HNSW_view =
        raft::make_host_matrix_view<uint64_t, int64_t>(indices_HNSW.data(), ps.n_queries, ps.k);
      auto distances_HNSW_view =
        raft::make_host_matrix_view<float, int64_t>(distances_HNSW.data(), ps.n_queries, ps.k);
      cuvs::neighbors::hnsw::search(handle_,
                                    search_params,
                                    *hnsw_index.get(),
                                    queries_HNSW_view,
                                    indices_HNSW_view,
                                    distances_HNSW_view);
    }

    double min_recall = ps.min_recall;
    EXPECT_TRUE(eval_neighbours(indices_naive,
                                indices_HNSW,
                                distances_naive,
                                distances_HNSW,
                                ps.n_queries,
                                ps.k,
                                0.006,
                                min_recall));
  }

  void SetUp() override
  {
    database.resize(((size_t)ps.n_rows) * ps.dim, stream_);
    queries.resize(((size_t)ps.n_queries) * ps.dim, stream_);
    raft::random::RngState r(1234ULL);
    if constexpr (std::is_same_v<DataT, float> || std::is_same_v<DataT, half>) {
      raft::random::uniform(
        handle_, r, database.data(), ps.n_rows * ps.dim, DataT(0.1), DataT(2.0));
      raft::random::uniform(
        handle_, r, queries.data(), ps.n_queries * ps.dim, DataT(0.1), DataT(2.0));
    } else {
      raft::random::uniformInt(
        handle_, r, database.data(), ps.n_rows * ps.dim, DataT(1), DataT(100));
      raft::random::uniformInt(
        handle_, r, queries.data(), ps.n_queries * ps.dim, DataT(1), DataT(100));
    }
    raft::resource::sync_stream(handle_);
  }

  void TearDown() override
  {
    raft::resource::sync_stream(handle_);
    database.resize(0, stream_);
    queries.resize(0, stream_);
  }

 private:
  raft::resources handle_;
  cuda::stream_ref stream_;
  AnnHNSWInputs ps;
  rmm::device_uvector<DataT> database;
  rmm::device_uvector<DataT> queries;
};

// Hold the CAGRA graph fixed so build randomness cannot mask conversion differences.
template <typename T, typename DistT>
void check_flat_conversion(cuvs::distance::DistanceType metric)
{
  raft::resources res;
  constexpr int64_t rows = 97, dim = 7, degree = 5, k = 4;
  auto vectors          = raft::make_host_matrix<T, int64_t>(rows, dim);
  auto override_vectors = raft::make_host_matrix<T, int64_t>(rows, dim);
  auto graph            = raft::make_pinned_matrix<uint32_t, int64_t>(res, rows, degree);
  std::fill_n(graph.data_handle(), rows * degree, cagra::kInvalidNeighbor<uint32_t>);
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < dim; ++j) {
      vectors(i, j)          = static_cast<T>((i * 23 + j * 11) % 103 - 51);
      override_vectors(i, j) = vectors(i, j) + 1;
    }
    const int64_t neighbor_count = 2 + i % (degree - 1);
    for (int64_t j = 0; j < neighbor_count; ++j) {
      graph(i, j) = static_cast<uint32_t>((i + j + 1) % rows);
    }
  }
  auto data          = raft::make_const_mdspan(vectors.view());
  auto override_data = raft::make_const_mdspan(override_vectors.view());
  index_params params;
  params.hierarchy   = HnswHierarchy::NONE;
  params.num_threads = 1;
  search_params search_options;
  search_options.ef          = 12;
  search_options.num_threads = 1;
  auto search_all            = [&](const index<T>& idx) {
    std::vector<uint64_t> ids(rows * k);
    std::vector<float> distances(rows * k);
    search(res,
           search_options,
           idx,
           data,
           raft::make_host_matrix_view(ids.data(), rows, k),
           raft::make_host_matrix_view(distances.data(), rows, k));
    return std::make_pair(std::move(ids), std::move(distances));
  };
  auto check_conversion = [&](const auto& cagra_index, std::optional<decltype(data)> host_vectors) {
    const auto expected = host_vectors.value_or(data);
    tmp_index_file reference, actual;
    cagra::serialize_to_hnswlib(res, reference.filename, cagra_index, host_vectors);
    auto direct = from_cagra(res, params, cagra_index, host_vectors);
    serialize(res, actual.filename, *direct);
    std::ifstream legacy_file(reference.filename, std::ios::binary),
      direct_file(actual.filename, std::ios::binary);
    ASSERT_TRUE(legacy_file && direct_file);
    using bytes = std::istreambuf_iterator<char>;
    EXPECT_TRUE(std::equal(bytes(legacy_file), bytes(), bytes(direct_file), bytes()));

    using native_index = hnswlib::HierarchicalNSW<DistT>;
    const auto* native = static_cast<const native_index*>(direct->get_index());
    ASSERT_TRUE(native->base_layer_only);
    for (int64_t i = 0; i < rows; ++i) {
      auto vec = native->template getDataByLabel<T>(i);
      ASSERT_EQ(vec.size(), dim);
      EXPECT_TRUE(std::equal(vec.begin(), vec.end(), &expected(i, 0)));
    }
    index<T>* loaded = nullptr;
    deserialize(res, params, reference.filename, dim, metric, &loaded);
    std::unique_ptr<index<T>> legacy(loaded);
    EXPECT_EQ(native->num_seeds, static_cast<const native_index*>(legacy->get_index())->num_seeds);
    EXPECT_EQ(search_all(*direct), search_all(*legacy));
  };
  // A device view of the pinned allocation gives CAGRA a graph that the CPU can read directly.
  auto mapped_graph =
    raft::make_device_matrix_view<const uint32_t, int64_t>(graph.data_handle(), rows, degree);
  cagra::host_standard_index<T> host_index(
    res, metric, make_host_standard_dataset_view(data), mapped_graph);
  check_conversion(host_index, data);
  check_conversion(host_index, override_data);

  // Passing a host view to the constructor makes CAGRA copy the graph to the GPU.
  auto host_graph =
    raft::make_host_matrix_view<const uint32_t, int64_t>(graph.data_handle(), rows, degree);
  auto padded = make_device_padded_dataset(res, data);
  cagra::device_padded_index<T> device_index(res, metric, padded->view(), host_graph);
  check_conversion(device_index, std::nullopt);
  check_conversion(device_index, override_data);
  device_index.update_graph(res, mapped_graph);
  check_conversion(device_index, std::nullopt);
}

TEST(AnnHNSW, FlatConversionMatchesHnswlibExport)
{
  check_flat_conversion<float, float>(cuvs::distance::DistanceType::L2Expanded);
  check_flat_conversion<int8_t, int>(cuvs::distance::DistanceType::InnerProduct);
}

TEST(AnnHNSW, FlatConversionPreservesPartialSecondBatch)
{
  raft::resources res;
  constexpr int64_t dim = 257, degree = 5;
  // Match the flat record layout and cross copy_flat_hnsw_rows' 64 MiB staging target.
  constexpr size_t row_bytes = sizeof(hnswlib::linklistsizeint) + degree * sizeof(uint32_t) +
                               dim * sizeof(float) + sizeof(hnswlib::labeltype);
  constexpr int64_t batch_rows = (64 * 1024 * 1024) / row_bytes;
  constexpr int64_t rows       = batch_rows + 17;
  auto vectors                 = raft::make_host_matrix<float, int64_t>(rows, dim);
  auto graph                   = raft::make_host_matrix<uint32_t, int64_t>(rows, degree);
  std::fill_n(graph.data_handle(), rows * degree, cagra::kInvalidNeighbor<uint32_t>);
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < dim; ++j) {
      // Distinct rows make a repeated first batch observable in the serialized output.
      vectors(i, j) = static_cast<float>(i * dim + j);
    }
    for (int64_t j = 0; j < 2 + i % (degree - 1); ++j) {
      graph(i, j) = static_cast<uint32_t>((i + j + 1) % rows);
    }
  }
  const auto data = raft::make_const_mdspan(vectors.view());
  auto padded     = make_device_padded_dataset(res, data);
  ASSERT_GT(padded->view().data_const_view().stride(0), dim);
  cagra::device_padded_index<float> cagra_index(res,
                                                cuvs::distance::DistanceType::L2Expanded,
                                                padded->view(),
                                                raft::make_const_mdspan(graph.view()));
  index_params params;
  params.hierarchy   = HnswHierarchy::NONE;
  params.num_threads = 4;

  tmp_index_file reference, actual;
  cagra::serialize_to_hnswlib(res, reference.filename, cagra_index);
  auto direct = from_cagra(res, params, cagra_index);
  serialize(res, actual.filename, *direct);
  std::ifstream legacy_file(reference.filename, std::ios::binary),
    direct_file(actual.filename, std::ios::binary);
  ASSERT_TRUE(legacy_file && direct_file);
  using bytes = std::istreambuf_iterator<char>;
  EXPECT_TRUE(std::equal(bytes(legacy_file), bytes(), bytes(direct_file), bytes()));

  const auto* native = static_cast<const hnswlib::HierarchicalNSW<float>*>(direct->get_index());
  ASSERT_EQ(native->size_data_per_element_, row_bytes);
  // Check label lookup and vectors at the boundary and throughout the partial tail.
  for (int64_t i = batch_rows - 1; i < rows; ++i) {
    SCOPED_TRACE(i);
    auto vec = native->getDataByLabel<float>(i);
    ASSERT_EQ(vec.size(), dim);
    EXPECT_TRUE(std::equal(vec.begin(), vec.end(), &data(i, 0)));
  }
}

const std::vector<AnnHNSWInputs> inputs = raft::util::itertools::product<AnnHNSWInputs>(
  {1000, 2000},                          // n_rows
  {5, 10, 25, 50, 100, 250, 500, 1000},  // dim
  {32, 64},                              // graph_degree
  {cuvs::distance::DistanceType::L2Expanded, cuvs::distance::DistanceType::InnerProduct},  // metric
  {50},                                                                                    // k
  {500},  // n_queries
  {250},  // ef
  {0.98}  // min_recall
);

typedef AnnHNSWTest<float, float, uint64_t> AnnHNSW_F;
TEST_P(AnnHNSW_F, AnnHNSW) { this->testHNSW(); }

INSTANTIATE_TEST_CASE_P(AnnHNSWTest, AnnHNSW_F, ::testing::ValuesIn(inputs));

typedef AnnHNSWTest<float, half, uint64_t> AnnHNSW_H;
TEST_P(AnnHNSW_H, AnnHNSW) { this->testHNSW(); }

INSTANTIATE_TEST_CASE_P(AnnHNSWTest, AnnHNSW_H, ::testing::ValuesIn(inputs));

typedef AnnHNSWTest<float, int8_t, uint64_t> AnnHNSW_I8;
TEST_P(AnnHNSW_I8, AnnHNSW) { this->testHNSW(); }

INSTANTIATE_TEST_CASE_P(AnnHNSWTest, AnnHNSW_I8, ::testing::ValuesIn(inputs));

typedef AnnHNSWTest<float, uint8_t, uint64_t> AnnHNSW_U8;
TEST_P(AnnHNSW_U8, AnnHNSW) { this->testHNSW(); }

INSTANTIATE_TEST_CASE_P(AnnHNSWTest, AnnHNSW_U8, ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::hnsw
