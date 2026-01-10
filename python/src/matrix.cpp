// Copyright 2023 BPCells contributors
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#include "matrix.hpp"
#include "py_interrupts.hpp"

#include <algorithm>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <random>
#include <sstream>
#include <iomanip>

#include <Eigen/SparseCore>
#include "bpcells-cpp/matrixIterators/StoredMatrixSparseColumn.h"

#include "bpcells-cpp/arrayIO/binaryfile.h"
#include "bpcells-cpp/arrayIO/vector.h"
#include "bpcells-cpp/utils/filesystem_compat.h"

#include "bpcells-cpp/matrixIterators/CSparseMatrix.h"
#include "bpcells-cpp/matrixIterators/MatrixIndexSelect.h"
#include "bpcells-cpp/matrixIterators/StoredMatrix.h"
#include "bpcells-cpp/matrixIterators/StoredMatrixWriter.h"
#include "bpcells-cpp/matrixIterators/ImportMatrix10xHDF5.h"
#include "bpcells-cpp/matrixIterators/ImportMatrixAnnDataHDF5.h"
#include "bpcells-cpp/matrixIterators/ConcatenateMatrix.h"


namespace BPCells::py {

void write_matrix_dir_from_memory(
    const Eigen::SparseMatrix<uint32_t> in, std::string out_path, bool row_major
) {
    const Eigen::Map<Eigen::SparseMatrix<uint32_t>> in_map(
        in.rows(),
        in.cols(),
        in.nonZeros(),
        (int *)in.outerIndexPtr(),
        (int *)in.innerIndexPtr(),
        (uint32_t *)in.valuePtr()
    );

    auto mat = std::make_unique<CSparseMatrix<uint32_t>>(in_map);

    FileWriterBuilder wb(out_path);

    run_with_py_interrupt_check(
        &StoredMatrixWriter<uint32_t>::write,
        StoredMatrixWriter<uint32_t>::createPacked(wb, row_major),
        std::ref(*mat)
    );
}

void write_matrix_dir_from_memory_experimental(
    const Eigen::SparseMatrix<uint32_t> in, std::string out_path
) {
    const Eigen::Map<Eigen::SparseMatrix<uint32_t>> in_map(
        in.rows(),
        in.cols(),
        in.nonZeros(),
        (int *)in.outerIndexPtr(),
        (int *)in.innerIndexPtr(),
        (uint32_t *)in.valuePtr()
    );

    auto mat = std::make_unique<CSparseMatrix<uint32_t>>(in_map);

    FileWriterBuilder wb(out_path);

    run_with_py_interrupt_check(
        &StoredMatrixWriter<uint32_t>::write,
        EXPERIMENTAL_createPackedSparseColumn<uint32_t>(wb),
        std::ref(*mat)
    );
}

static bool is_row_major_matrix_dir(std::string path) {
    FileReaderBuilder rb(path);
    auto storage_order_reader = rb.openStringReader("storage_order");
    auto storage_order = storage_order_reader->get(0);
    bool row_major = false;
    if (std::string_view("row") == storage_order) row_major = true;
    else if (std::string("col") == storage_order) row_major = false;
    else
        throw std::runtime_error(
            std::string("storage_order must be either \"row\" or \"col\", found: \"") +
            storage_order + "\""
        );
    return row_major;
}

void write_matrix_dir_from_concat(std::vector<std::string> in_paths, std::string out_path, bool concat_cols)  {
    std::vector<std::unique_ptr<MatrixLoader<uint32_t>>> mats;

    if (in_paths.size() == 0) {
        throw std::runtime_error("write_matrix_dir_from_hstack: Zero matrices given as input.");
    }

    bool row_major = is_row_major_matrix_dir(in_paths[0]);

    for (const std::string &path : in_paths) {
        FileReaderBuilder rb(path);
        mats.push_back(std::make_unique<StoredMatrix<uint32_t>>(StoredMatrix<uint32_t>::openPacked(rb)));
    }

    std::unique_ptr<MatrixLoader<uint32_t>> mat;
    if (row_major == concat_cols) {
        mat = std::make_unique<ConcatRows<uint32_t>>(std::move(mats), 0);
    } else {
        mat = std::make_unique<ConcatCols<uint32_t>>(std::move(mats), 0);
    }

    FileWriterBuilder wb(out_path);

    run_with_py_interrupt_check(
        &StoredMatrixWriter<uint32_t>::write,
        StoredMatrixWriter<uint32_t>::createPacked(wb, row_major),
        std::ref(*mat)
    );
}

// Helper to generate a unique temp directory name
static std::string generate_unique_id() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint64_t> dis;
    std::stringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(16) << dis(gen);
    return ss.str();
}

// Internal helper for merging a batch of experimental matrices (single-threaded)
static void merge_batch_experimental_internal(
    const std::vector<std::string>& in_paths,
    const std::string& out_path,
    bool concat_rows,
    std::atomic<bool>* user_interrupt
) {
    std::vector<std::unique_ptr<MatrixLoader<uint32_t>>> mats;

    for (const std::string &path : in_paths) {
        FileReaderBuilder rb(path);
        mats.push_back(std::make_unique<StoredMatrix<uint32_t>>(EXPERIMENTAL_openPackedSparseColumn<uint32_t>(rb)));
    }

    std::unique_ptr<MatrixLoader<uint32_t>> mat;
    if (concat_rows) {
        mat = std::make_unique<ConcatRows<uint32_t>>(std::move(mats), 0);
    } else {
        mat = std::make_unique<ConcatCols<uint32_t>>(std::move(mats), 0);
    }

    FileWriterBuilder wb(out_path);
    auto writer = EXPERIMENTAL_createPackedSparseColumn<uint32_t>(wb);

    // ConcatRows produces sorted output (row offsets create non-overlapping, increasing ranges)
    // so we can skip the sorting check for better performance
    if (concat_rows) {
        writer.writeSorted(*mat, user_interrupt);
    } else {
        writer.write(*mat, user_interrupt);
    }
}

// Simple concat for small number of matrices
static void write_matrix_dir_from_concat_experimental_simple(
    const std::vector<std::string>& in_paths,
    const std::string& out_path,
    bool concat_rows
) {
    std::vector<std::unique_ptr<MatrixLoader<uint32_t>>> mats;

    for (const std::string &path : in_paths) {
        FileReaderBuilder rb(path);
        mats.push_back(std::make_unique<StoredMatrix<uint32_t>>(EXPERIMENTAL_openPackedSparseColumn<uint32_t>(rb)));
    }

    std::unique_ptr<MatrixLoader<uint32_t>> mat;
    if (concat_rows) {
        mat = std::make_unique<ConcatRows<uint32_t>>(std::move(mats), 0);
    } else {
        mat = std::make_unique<ConcatCols<uint32_t>>(std::move(mats), 0);
    }

    FileWriterBuilder wb(out_path);

    // ConcatRows produces sorted output, so use writeSorted for better performance
    if (concat_rows) {
        run_with_py_interrupt_check(
            &StoredMatrixWriter<uint32_t>::writeSorted,
            EXPERIMENTAL_createPackedSparseColumn<uint32_t>(wb),
            std::ref(*mat)
        );
    } else {
        run_with_py_interrupt_check(
            &StoredMatrixWriter<uint32_t>::write,
            EXPERIMENTAL_createPackedSparseColumn<uint32_t>(wb),
            std::ref(*mat)
        );
    }
}

void write_matrix_dir_from_concat_experimental(
    std::vector<std::string> in_paths,
    std::string out_path,
    bool concat_rows,
    uint32_t batch_size,
    uint32_t threads,
    std::string temp_dir
) {
    // Concatenate experimental format matrices (packed-uint-matrix-v9999)
    // Used for precalculated pseudobulk coverage matrices
    // Supports hierarchical merge for better performance with many matrices

    if (in_paths.size() == 0) {
        throw std::runtime_error("write_matrix_dir_from_concat_experimental: Zero matrices given as input.");
    }

    if (in_paths.size() == 1) {
        // Just copy the single matrix
        write_matrix_dir_from_concat_experimental_simple(in_paths, out_path, concat_rows);
        return;
    }

    // If we have few enough matrices, use the simple merge
    if (in_paths.size() <= batch_size) {
        write_matrix_dir_from_concat_experimental_simple(in_paths, out_path, concat_rows);
        return;
    }

    // Set up temp directory
    std_fs::path temp_base;
    bool created_temp_dir = false;
    if (temp_dir.empty()) {
        std_fs::path out_parent = std_fs::path(out_path).parent_path();
        if (out_parent.empty()) out_parent = ".";
        temp_base = out_parent / (".bpcells_tmp_" + generate_unique_id());
    } else {
        temp_base = std_fs::path(temp_dir);
    }

    if (!std_fs::exists(temp_base)) {
        std_fs::create_directories(temp_base);
        created_temp_dir = true;
    }

    // RAII cleanup helper
    struct TempDirCleanup {
        std_fs::path path;
        bool should_remove;
        ~TempDirCleanup() {
            if (should_remove && std_fs::exists(path)) {
                std::error_code ec;
                std_fs::remove_all(path, ec);
            }
        }
    };
    TempDirCleanup cleanup{temp_base, created_temp_dir};

    // Split paths into batches
    std::vector<std::vector<std::string>> batches;
    size_t n = in_paths.size();
    size_t num_batches = (n + batch_size - 1) / batch_size;

    for (size_t i = 0; i < num_batches; i++) {
        size_t start = i * batch_size;
        size_t end = std::min(start + batch_size, n);
        batches.emplace_back(in_paths.begin() + start, in_paths.begin() + end);
    }

    // Merge batches to intermediate files
    std::vector<std::string> intermediate_paths;

    if (threads <= 1) {
        // Single-threaded: merge batches sequentially
        run_with_py_interrupt_check([&](std::atomic<bool>* user_interrupt) {
            for (size_t i = 0; i < batches.size(); i++) {
                if (user_interrupt && *user_interrupt) return;

                std::string intermediate_path = (temp_base / ("batch_" + std::to_string(i))).string();
                merge_batch_experimental_internal(batches[i], intermediate_path, concat_rows, user_interrupt);
                intermediate_paths.push_back(intermediate_path);
            }
        });
    } else {
        // Multi-threaded: merge batches in parallel
        intermediate_paths.resize(batches.size());
        for (size_t i = 0; i < batches.size(); i++) {
            intermediate_paths[i] = (temp_base / ("batch_" + std::to_string(i))).string();
        }

        run_with_py_interrupt_check([&](std::atomic<bool>* user_interrupt) {
            std::vector<std::future<void>> futures;
            std::atomic<size_t> task_id(0);

            // Create tasks using deferred execution
            for (size_t i = 0; i < batches.size(); i++) {
                futures.push_back(std::async(
                    std::launch::deferred,
                    [&batches, &intermediate_paths, concat_rows, i, user_interrupt]() {
                        merge_batch_experimental_internal(batches[i], intermediate_paths[i], concat_rows, user_interrupt);
                    }
                ));
            }

            // Execute with thread pool
            uint32_t actual_threads = std::min(threads, static_cast<uint32_t>(batches.size()));
            std::vector<std::thread> thread_vec;

            for (uint32_t t = 0; t < actual_threads; t++) {
                thread_vec.push_back(std::thread([&futures, &task_id, user_interrupt] {
                    while (true) {
                        if (user_interrupt && *user_interrupt) break;
                        size_t cur_task = task_id.fetch_add(1);
                        if (cur_task >= futures.size()) break;
                        futures[cur_task].get();
                    }
                }));
            }

            for (auto& th : thread_vec) {
                if (th.joinable()) th.join();
            }
        });
    }

    // Recursively merge intermediate files
    if (intermediate_paths.size() <= batch_size) {
        // Final merge directly to output
        write_matrix_dir_from_concat_experimental_simple(intermediate_paths, out_path, concat_rows);
    } else {
        // Need another level of recursion
        std::string next_temp_dir = (temp_base / "level2").string();
        write_matrix_dir_from_concat_experimental(
            intermediate_paths, out_path, concat_rows, batch_size, threads, next_temp_dir
        );
    }
}

void write_matrix_dir_from_h5ad(std::string h5ad_path, std::string out_path, std::string group) {
    std::vector<std::string> empty_names;
    auto row_names = std::make_unique<VecStringReader>(empty_names);
    auto col_names = std::make_unique<VecStringReader>(empty_names);

    // Get the AnnData matrix directly
    auto mat_float = openAnnDataMatrix<float>(
        h5ad_path,
        group,
        16384L, // Just provide a default buffer size matching what R uses
        std::move(row_names),
        std::move(col_names)
    );
    auto mat_int = std::make_unique<MatrixConverterLoader<float, uint32_t>>(std::move(mat_float));
    
    bool row_major = isRowOrientedAnnDataMatrix(h5ad_path, group);
    
    FileWriterBuilder wb(out_path);

    run_with_py_interrupt_check(
        &StoredMatrixWriter<uint32_t>::write,
        StoredMatrixWriter<uint32_t>::createPacked(wb, row_major),
        std::ref(*mat_int)
    );
}

template <typename T>
std::vector<T> parallel_map_helper(std::vector<std::future<T>> &futures, size_t threads) {
    std::vector<T> result(futures.size());

    // Non-threaded fallback
    if (threads == 0) {
        for (size_t i = 0; i < futures.size(); i++) {
            result[i] = futures[i].get();
        }
        return result;
    }

    // Very basic threading, designed for small numbers of futures
    std::atomic<size_t> task_id(0);
    std::vector<std::thread> thread_vec;
    for (size_t i = 0; i < threads; i++) {
        thread_vec.push_back(std::thread([&futures, &result, &task_id] {
            while (true) {
                size_t cur_task = task_id.fetch_add(1);
                if (cur_task >= futures.size()) break;
                result[cur_task] = futures[cur_task].get();
            }
        }));
    }
    for (auto &th : thread_vec) {
        if (th.joinable()) {
            th.join();
        }
    }
    return result;
}

VecReaderWriterBuilder load_matrix_dir_to_memory(std::string matrix_path) {
    FileReaderBuilder rb(matrix_path);
    std::unique_ptr<MatrixLoader<uint32_t>> mat =
        std::make_unique<StoredMatrix<uint32_t>>(StoredMatrix<uint32_t>::openPacked(rb));

    VecReaderWriterBuilder wb;
    run_with_py_interrupt_check(
        &StoredMatrixWriter<uint32_t>::write,
        StoredMatrixWriter<uint32_t>::createPacked(wb),
        std::ref(*mat)
    );
    return wb;
}

Eigen::SparseMatrix<uint32_t> load_matrix_subset_helper(
    ReaderBuilder &rb,
    std::optional<std::vector<uint32_t>> rows,
    std::vector<uint32_t> columns,
    std::atomic<bool> *user_interrupt
) {
    std::unique_ptr<MatrixLoader<uint32_t>> mat;
    
    // Check version - experimental matrices use version 9999
    // Note: This only works for FileReaderBuilder, but that's what we use for load_matrix_dir_subset
    try {
        std::string version = rb.readVersion();
        if (version == StoredMatrix<uint32_t>::versionString(true, 9999)) {
            // Use experimental reader for version 9999
            mat = std::make_unique<StoredMatrix<uint32_t>>(EXPERIMENTAL_openPackedSparseColumn<uint32_t>(rb));
        } else {
            // Use standard reader for version 2
            mat = std::make_unique<StoredMatrix<uint32_t>>(StoredMatrix<uint32_t>::openPacked(rb));
        }
    } catch (...) {
        // If readVersion() fails (e.g., for VecReaderWriterBuilder), fall back to standard reader
        mat = std::make_unique<StoredMatrix<uint32_t>>(StoredMatrix<uint32_t>::openPacked(rb));
    }

    mat = std::make_unique<MatrixColSelect<uint32_t>>(std::move(mat), columns);
    if (rows) {
        mat = std::make_unique<MatrixRowSelect<uint32_t>>(std::move(mat), rows.value());
    }

    CSparseMatrixWriter<uint32_t> writer;
    run_with_py_interrupt_check(
        &CSparseMatrixWriter<uint32_t>::write, std::ref(writer), std::ref(*mat)
    );
    return writer.getMat();
}

std::vector<Eigen::SparseMatrix<uint32_t>> load_matrix_subset(
    ReaderBuilder &rb,
    std::optional<std::vector<uint32_t>> rows,
    std::vector<uint32_t> columns,
    uint32_t threads
) {
    // Split columns into chunks
    std::vector<std::vector<uint32_t>> col_splits;
    uint32_t chunks = std::max<uint32_t>(1, threads);
    uint32_t idx = 0;
    for (uint32_t i = 0; i < chunks; i++) {
        std::vector<uint32_t> c;
        uint32_t col_count = (columns.size() - idx) / (chunks - i);
        for (uint32_t j = 0; j < col_count; j++) {
            c.push_back(columns[idx]);
            idx++;
        }
        col_splits.push_back(c);
    }

    // Perform multi-threaded reads
    return run_with_py_interrupt_check(
        [&rb, &rows, &col_splits, threads](std::atomic<bool> *user_interrupt) {
            std::vector<std::future<Eigen::SparseMatrix<uint32_t>>> task_vec;
            for (size_t i = 0; i < col_splits.size(); i++) {
                task_vec.push_back(std::async(
                    std::launch::deferred,
                    &load_matrix_subset_helper,
                    std::ref(rb),
                    rows,
                    col_splits[i],
                    user_interrupt
                ));
            }

            return parallel_map_helper(task_vec, threads);
        }
    );
}

std::vector<Eigen::SparseMatrix<uint32_t>> load_matrix_dir_subset(
    std::string matrix_path,
    std::optional<std::vector<uint32_t>> rows,
    std::vector<uint32_t> columns,
    uint32_t threads
) {
    FileReaderBuilder rb(matrix_path);
    return load_matrix_subset(rb, rows, columns, threads);
}

std::vector<Eigen::SparseMatrix<uint32_t>> load_matrix_memory_subset(
    VecReaderWriterBuilder &rb,
    std::optional<std::vector<uint32_t>> rows,
    std::vector<uint32_t> columns,
    uint32_t threads
) {
    return load_matrix_subset(rb, rows, columns, threads);
}

std::tuple<uint32_t, uint32_t> dims_matrix_dir(std::string matrix_path) {
    FileReaderBuilder rb(matrix_path);
    std::unique_ptr<MatrixLoader<uint32_t>> mat;
    
    // Check version - experimental matrices use version 9999
    std::string version = rb.readVersion();
    if (version == StoredMatrix<uint32_t>::versionString(true, 9999)) {
        // Use experimental reader for version 9999
        mat = std::make_unique<StoredMatrix<uint32_t>>(EXPERIMENTAL_openPackedSparseColumn<uint32_t>(rb));
    } else {
        // Use standard reader for version 2
        mat = std::make_unique<StoredMatrix<uint32_t>>(StoredMatrix<uint32_t>::openPacked(rb));
    }

    return {mat->rows(), mat->cols()};
}

std::vector<std::string> row_names_stored_matrix(std::string matrix_path) {
    FileReaderBuilder rb(matrix_path);
    std::unique_ptr<MatrixLoader<uint32_t>> mat =
        std::make_unique<StoredMatrix<uint32_t>>(EXPERIMENTAL_openPackedSparseColumn<uint32_t>(rb));

    std::vector<std::string> row_names;
    for (uint32_t i = 0; i < mat->rows(); i++) {
        const char* name = mat->rowNames(i);
        if (name != nullptr) {
            row_names.push_back(std::string(name));
        } else {
            row_names.push_back("");
        }
    }
    return row_names;
}

} // namespace BPCells::py