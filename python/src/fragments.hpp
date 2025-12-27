// Copyright 2023 BPCells contributors
// 
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#pragma once

#include <atomic>
#include <exception>
#include <future>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Core>

namespace BPCells::py {

void import_10x_fragments(
    std::string input_10x,
    std::string output_bpcells,
    int shift_start,
    int shift_end,
    std::optional<std::vector<std::string>> keeper_cells
);

std::vector<uint32_t> echo_vec(std::vector<uint32_t> a);

std::vector<std::string> cell_names_fragments_dir(std::string input_bpcells);
std::vector<std::string> chr_names_fragments_dir(std::string input_bpcells);

Eigen::MatrixXi pseudobulk_coverage(
    std::string fragments_path,
    std::vector<std::string> chr,
    std::vector<uint32_t> start,
    std::vector<uint32_t> end,
    std::vector<int32_t> cell_groups,
    int bin_size
);

// Calculate an experimental CompressedSparseColumn matrix
void precalculate_pseudobulk_coverage(
    std::vector<std::string> fragments_paths,
    std::string output_path,
    std::string tmp_path,
    std::vector<std::string> chr,
    std::vector<uint32_t> chr_size,
    std::vector<int32_t> cell_groups,
    int bin_size,
    int threads,
    std::optional<std::vector<std::string>> group_names = std::nullopt
);


Eigen::MatrixXi query_precalculated_pseudobulk_coverage(
    std::string mat_path,
    std::vector<uint32_t> range_starts,
    uint32_t range_len
);

// Template helper for parallel execution of tasks
template <typename T>
void parallel_map_helper(std::vector<std::future<T>> &futures, size_t threads, std::vector<T> *results = nullptr) {
    // Non-threaded fallback
    if (threads == 0) {
        for (size_t i = 0; i < futures.size(); i++) {
            if (results) {
                (*results)[i] = futures[i].get();
            } else {
                futures[i].get();
            }
        }
        return;
    }

    // Very basic threading, designed for small numbers of futures
    std::atomic<size_t> task_id(0);
    std::atomic<bool> has_error = false;
    std::exception_ptr exception;
    std::vector<std::thread> thread_vec;
    for (size_t i = 0; i < threads; i++) {
        thread_vec.push_back(std::thread([&futures, &task_id, &has_error, &exception, results] {
            while (true) {
                size_t cur_task = task_id.fetch_add(1);
                if (cur_task >= futures.size()) break;
                try {
                    if (results) {
                        (*results)[cur_task] = futures[cur_task].get();
                    } else {
                        futures[cur_task].get();
                    }
                } catch (...) {
                    has_error = true;
                    exception = std::current_exception();
                    break;
                }
            }
        }));
    }
    for (auto &t : thread_vec) {
        t.join();
    }
    if (has_error) {
        std::rethrow_exception(exception);
    }
}

} // namespace BPCells::py