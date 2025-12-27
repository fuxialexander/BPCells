// Copyright 2023 BPCells contributors
// 
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#include "bam.hpp"
#include "fragments.hpp"  // For parallel_map_helper and other shared utilities
#include "py_interrupts.hpp"

#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <future>
#include <thread>
#include <unordered_map>
#include <algorithm>
#include <atomic>

#include "bpcells-cpp/arrayIO/binaryfile.h"
#include "bpcells-cpp/arrayIO/vector.h"
#include "bpcells-cpp/fragmentIterators/BamFragments.h"
#include "bpcells-cpp/fragmentIterators/CellSelect.h"
#include "bpcells-cpp/fragmentIterators/MergeFragments.h"
#include "bpcells-cpp/fragmentIterators/ShiftCoords.h"
#include "bpcells-cpp/matrixIterators/ConcatenateMatrix.h"
#include "bpcells-cpp/matrixIterators/MatrixIndexSelect.h"
#include "bpcells-cpp/matrixIterators/StoredMatrixSparseColumn.h"
#include "bpcells-cpp/matrixIterators/StoredMatrixWriter.h"
#include "bpcells-cpp/matrixIterators/RenameDims.h"
#include "bpcells-cpp/matrixIterators/TileMatrix.h"
#include "bpcells-cpp/utils/filesystem_compat.h"

namespace BPCells::py {

// Write a chunk of the tile matrix columns to the given output path (BAM version)
// This helper MUST instantiate BamFragments locally inside the thread for thread safety
static std::vector<uint64_t> precalculate_pseudobulk_coverage_bam_helper(
    std::string bam_path,
    std::string chunk_output_path,
    std::pair<uint32_t, uint32_t> chunk_col_range,

    const std::vector<uint32_t> &group_ids,
    const std::vector<std::string> &group_names,

    const std::vector<uint32_t> &chr_id,
    const std::vector<uint32_t> &start,
    const std::vector<uint32_t> &end,
    const std::vector<uint32_t> &width,
    const std::vector<std::string> &chr_levels,
    
    int bin_size,
    int shift_start,
    int shift_end,

    std::atomic<bool> *user_interrupt
) {
    // 1. Open BAM file locally (thread-safe - each thread gets its own handle)
    std::unique_ptr<FragmentLoader> frags = std::make_unique<BamFragments>(bam_path);

    // 2. Pre-scan to discover all cells (needed for CellMerge validation)
    // We need to read through the file once to discover all cells before creating CellMerge
    // This ensures cellCount() returns the correct value
    frags->restart();
    while (frags->nextChr()) {
        while (frags->load()) {
            // Just load to discover cells, don't need to process data
        }
    }
    // Restart again for actual processing
    frags->restart();

    // 3. Apply Tn5 Shift if needed
    if (shift_start != 0 || shift_end != 0) {
        frags = std::make_unique<ShiftCoords>(std::move(frags), shift_start, shift_end);
    }

    // 4. Verify cell count matches before creating CellMerge
    int discovered_cell_count = frags->cellCount();
    if (discovered_cell_count != (int)group_ids.size()) {
        throw std::runtime_error(
            "CellMerge cell count mismatch: discovered " + std::to_string(discovered_cell_count) +
            " cells in BAM, but cell_groups array has length " + std::to_string(group_ids.size()) +
            ". This usually means cells were discovered in a different order than expected."
        );
    }

    // 5. Merge cells (now cellCount() will return the correct value)
    frags = std::make_unique<CellMerge>(
        std::move(frags), group_ids, std::make_unique<VecStringReader>(group_names)
    );

    // 6. Construct tile matrix
    std::unique_ptr<MatrixLoader<uint32_t>> tile_mat = std::make_unique<TileMatrix>(
        std::move(frags), chr_id, start, end, width, std::make_unique<VecStringReader>(chr_levels), false
    );

    // 5. Subset to the desired columns
    tile_mat = std::make_unique<MatrixColSlice<uint32_t>>(
        std::move(tile_mat), chunk_col_range.first, chunk_col_range.second
    );

    // 6. Track per-group sums in this chunk using efficient rowSums
    std::vector<uint64_t> group_sums(group_names.size(), 0);
    
    // Use the efficient rowSums implementation from MatrixOps.h
    std::vector<uint32_t> row_sums = tile_mat->rowSums(user_interrupt);
    
    // Convert to uint64_t and ensure we don't exceed group bounds
    for (size_t i = 0; i < row_sums.size() && i < group_sums.size(); i++) {
        group_sums[i] = static_cast<uint64_t>(row_sums[i]);
    }

    // 7. Reload fragments for writing (need to recreate since we moved it)
    std::unique_ptr<FragmentLoader> frags_new = std::make_unique<BamFragments>(bam_path);
    
    if (shift_start != 0 || shift_end != 0) {
        frags_new = std::make_unique<ShiftCoords>(std::move(frags_new), shift_start, shift_end);
    }
    
    auto merged_frags = std::make_unique<CellMerge>(
        std::move(frags_new),
        group_ids,
        std::make_unique<VecStringReader>(group_names)
    );
    
    // 8. Properly construct TileMatrix
    tile_mat = std::make_unique<TileMatrix>(
        std::move(merged_frags), 
        chr_id,
        start,
        end,
        width,
        std::make_unique<VecStringReader>(chr_levels),
        false
    );
    
    // Apply the same column slice
    tile_mat = std::make_unique<MatrixColSlice<uint32_t>>(
        std::move(tile_mat), chunk_col_range.first, chunk_col_range.second
    );

    // 9. Clear the row/col names
    std::vector<std::string> empty;
    tile_mat = std::make_unique<RenameDims<uint32_t>>(
        std::move(tile_mat), empty, empty, true, true
    );

    // 10. Write to output
    // Use version 9999 (experimental) if bin_size == 1, otherwise use version 2 (standard)
    FileWriterBuilder wb(chunk_output_path);
    if (bin_size == 1) {
        EXPERIMENTAL_createPackedSparseColumn<uint32_t>(wb).write(*tile_mat, user_interrupt);
    } else {
        StoredMatrixWriter<uint32_t>::createPacked(wb).write(*tile_mat, user_interrupt);
    }
    
    return group_sums;
}

// Write a chunk of the tile matrix columns to the given output path (Multi-BAM version)
// This helper MUST instantiate BamFragments locally inside the thread for thread safety
static std::vector<uint64_t> precalculate_pseudobulk_coverage_bam_multi_helper(
    const std::vector<std::string> &bam_paths,
    const std::vector<std::string> &bam_prefixes,  // Cell prefixes for each BAM
    std::string chunk_output_path,
    std::pair<uint32_t, uint32_t> chunk_col_range,

    const std::vector<uint32_t> &group_ids,
    const std::vector<std::string> &group_names,

    const std::vector<uint32_t> &chr_id,
    const std::vector<uint32_t> &start,
    const std::vector<uint32_t> &end,
    const std::vector<uint32_t> &width,
    const std::vector<std::string> &chr_levels,
    
    int bin_size,
    int shift_start,
    int shift_end,

    std::atomic<bool> *user_interrupt
) {
    // 1. Create BamFragments loaders for each BAM file with unique cell prefixes
    std::vector<std::unique_ptr<FragmentLoader>> frag_loaders;
    for (size_t i = 0; i < bam_paths.size(); i++) {
        std::string prefix = (i < bam_prefixes.size()) ? bam_prefixes[i] : "";
        frag_loaders.push_back(
            std::make_unique<BamFragments>(bam_paths[i], "CB", prefix)
        );
    }

    // 2. Merge all BAM fragments using MergeFragments
    std::unique_ptr<FragmentLoader> frags;
    if (frag_loaders.size() == 1) {
        frags = std::move(frag_loaders[0]);
    } else {
        frags = std::make_unique<MergeFragments>(std::move(frag_loaders), chr_levels);
    }

    // 3. Pre-scan to discover all cells (needed for CellMerge validation)
    frags->restart();
    while (frags->nextChr()) {
        while (frags->load()) {
            // Just load to discover cells, don't need to process data
        }
    }
    // Restart again for actual processing
    frags->restart();

    // 4. Apply Tn5 Shift if needed
    if (shift_start != 0 || shift_end != 0) {
        frags = std::make_unique<ShiftCoords>(std::move(frags), shift_start, shift_end);
    }

    // 5. Verify cell count matches before creating CellMerge
    int discovered_cell_count = frags->cellCount();
    if (discovered_cell_count != (int)group_ids.size()) {
        throw std::runtime_error(
            "CellMerge cell count mismatch: discovered " + std::to_string(discovered_cell_count) +
            " cells across all BAM files, but cell_groups array has length " + std::to_string(group_ids.size()) +
            ". This usually means cells were discovered in a different order than expected."
        );
    }

    // 6. Merge cells (now cellCount() will return the correct value)
    frags = std::make_unique<CellMerge>(
        std::move(frags), group_ids, std::make_unique<VecStringReader>(group_names)
    );

    // 7. Construct tile matrix
    std::unique_ptr<MatrixLoader<uint32_t>> tile_mat = std::make_unique<TileMatrix>(
        std::move(frags), chr_id, start, end, width, std::make_unique<VecStringReader>(chr_levels), false
    );

    // 8. Subset to the desired columns
    tile_mat = std::make_unique<MatrixColSlice<uint32_t>>(
        std::move(tile_mat), chunk_col_range.first, chunk_col_range.second
    );

    // 9. Track per-group sums in this chunk using efficient rowSums
    std::vector<uint64_t> group_sums(group_names.size(), 0);
    
    // Use the efficient rowSums implementation from MatrixOps.h
    std::vector<uint32_t> row_sums = tile_mat->rowSums(user_interrupt);
    
    // Convert to uint64_t and ensure we don't exceed group bounds
    for (size_t i = 0; i < row_sums.size() && i < group_sums.size(); i++) {
        group_sums[i] = static_cast<uint64_t>(row_sums[i]);
    }

    // 10. Reload fragments for writing (need to recreate since we moved it)
    std::vector<std::unique_ptr<FragmentLoader>> frag_loaders_new;
    for (size_t i = 0; i < bam_paths.size(); i++) {
        std::string prefix = (i < bam_prefixes.size()) ? bam_prefixes[i] : "";
        frag_loaders_new.push_back(
            std::make_unique<BamFragments>(bam_paths[i], "CB", prefix)
        );
    }
    
    std::unique_ptr<FragmentLoader> frags_new;
    if (frag_loaders_new.size() == 1) {
        frags_new = std::move(frag_loaders_new[0]);
    } else {
        frags_new = std::make_unique<MergeFragments>(std::move(frag_loaders_new), chr_levels);
    }
    
    if (shift_start != 0 || shift_end != 0) {
        frags_new = std::make_unique<ShiftCoords>(std::move(frags_new), shift_start, shift_end);
    }
    
    auto merged_frags = std::make_unique<CellMerge>(
        std::move(frags_new),
        group_ids,
        std::make_unique<VecStringReader>(group_names)
    );
    
    // 11. Properly construct TileMatrix
    tile_mat = std::make_unique<TileMatrix>(
        std::move(merged_frags), 
        chr_id,
        start,
        end,
        width,
        std::make_unique<VecStringReader>(chr_levels),
        false
    );
    
    // Apply the same column slice
    tile_mat = std::make_unique<MatrixColSlice<uint32_t>>(
        std::move(tile_mat), chunk_col_range.first, chunk_col_range.second
    );

    // 12. Clear the row/col names
    std::vector<std::string> empty;
    tile_mat = std::make_unique<RenameDims<uint32_t>>(
        std::move(tile_mat), empty, empty, true, true
    );

    // 13. Write to output
    // Use version 9999 (experimental) if bin_size == 1, otherwise use version 2 (standard)
    FileWriterBuilder wb(chunk_output_path);
    if (bin_size == 1) {
        EXPERIMENTAL_createPackedSparseColumn<uint32_t>(wb).write(*tile_mat, user_interrupt);
    } else {
        StoredMatrixWriter<uint32_t>::createPacked(wb).write(*tile_mat, user_interrupt);
    }
    
    return group_sums;
}

void precalculate_pseudobulk_coverage_bam(
    std::string bam_path,
    std::string output_path,
    std::string tmp_path,
    std::vector<std::string> chr,
    std::vector<uint32_t> chr_len,
    std::vector<int32_t> cell_groups,
    int shift_start,
    int shift_end,
    int bin_size,
    int threads,
    std::optional<std::vector<std::string>> group_names
) {
    // Validate inputs
    if (chr.size() != chr_len.size()) {
        throw std::runtime_error(
            "precalculate_pseudobulk_coverage_bam: chr must be same length as chr_len"
        );
    }

    // Create the arguments needed for TileMatrix: start, tile_width, chr_id, chr_levels
    std::vector<uint32_t> start(chr.size(), 0);
    std::vector<uint32_t> tile_width(chr.size(), bin_size);

    // Open BAM to get chromosome names
    BamFragments bam_frags(bam_path);
    std::vector<uint32_t> chr_id;
    std::unordered_map<std::string, uint32_t> chr_name_lookup;
    std::vector<std::string> chr_levels;
    
    int bam_chr_count = bam_frags.chrCount();
    for (int32_t i = 0; i < bam_chr_count; i++) {
        const char *chr_name = bam_frags.chrNames(i);
        if (chr_name == nullptr) {
            throw std::runtime_error("precalculate_pseudobulk_coverage_bam: missing chr names in BAM");
        }
        chr_name_lookup[std::string(chr_name)] = i;
        chr_levels.push_back(std::string(chr_name));
    }
    
    for (auto &c : chr) {
        if (chr_name_lookup.find(c) == chr_name_lookup.end()) {
            throw std::runtime_error("precalculate_pseudobulk_coverage_bam: chromosome " + c + " not found in BAM");
        }
        chr_id.push_back(chr_name_lookup[c]);
    }

    // Create the arguments needed for CellMerge (group_id, group_names)
    uint32_t num_groups = 1 + *std::max_element(cell_groups.cbegin(), cell_groups.cend());

    std::vector<std::string> actual_group_names;
    if (group_names.has_value()) {
        actual_group_names = group_names.value();
        // Make sure we have enough names, fill with defaults if needed
        if (actual_group_names.size() < num_groups) {
            for (size_t i = actual_group_names.size(); i < num_groups; i++) {
                actual_group_names.push_back(std::to_string(i));
            }
        }
        // Add one more for the discard group
        actual_group_names.push_back("discard");
    } else {
        // Generate dummy names if not provided
        for (uint32_t i = 0; i < num_groups + 1; i++) {
            actual_group_names.push_back(std::to_string(i));
        }
    }

    std::vector<uint32_t> cell_groups_uint;
    for (const auto &x : cell_groups) {
        cell_groups_uint.push_back(x >= 0 ? x : num_groups);
    }

    // Split columns into chunks
    size_t total_columns = 0;
    for (const auto &x : chr_len) {
        total_columns += (x + bin_size - 1) / bin_size;  // Ceiling division
    }
    uint32_t chunks = std::max<uint32_t>(1, threads * 4);
    std::vector<std::pair<uint32_t,uint32_t>> chunk_col_splits;
    uint32_t idx = 0;
    for (uint32_t i = 0; i < chunks; i++) {
        uint32_t col_count = (total_columns - idx) / (chunks - i);
        chunk_col_splits.push_back({idx, idx + col_count});
        idx += col_count;
    }

    std::vector<std::string> chunk_output_paths;
    for (uint32_t i = 0; i < chunks; i++) {
        chunk_output_paths.push_back((std_fs::path(tmp_path) / std::to_string(i)).string());
    }

    // Vector to store group sums from all chunks
    std::vector<std::vector<uint64_t>> all_group_sums;
    const std::vector<std::string> &group_names_ref = actual_group_names;

    // Make all the matrix chunks
    run_with_py_interrupt_check([&bam_path,
                                        &chunk_output_paths,
                                        &chunk_col_splits,
                                        &cell_groups_uint,
                                        &group_names_ref,
                                        &chr_id,
                                        &start,
                                        &chr_len,
                                        &tile_width,
                                        &chr_levels,
                                        &all_group_sums,
                                        threads,
                                        chunks,
                                        bin_size,
                                        shift_start,
                                        shift_end](std::atomic<bool> *user_interrupt) {
        std::vector<std::future<std::vector<uint64_t>>> task_vec;
        all_group_sums.resize(chunks);

        for (size_t i = 0; i < chunks; i++) {
            task_vec.push_back(std::async(
                std::launch::deferred,
                &precalculate_pseudobulk_coverage_bam_helper,
                bam_path,
                chunk_output_paths[i],
                chunk_col_splits[i],

                std::cref(cell_groups_uint),
                std::cref(group_names_ref),

                std::cref(chr_id),
                std::cref(start),
                std::cref(chr_len),
                std::cref(tile_width),
                std::cref(chr_levels),
                bin_size,
                shift_start,
                shift_end,

                user_interrupt
            ));
        }

        // Process the futures and collect the group sums in parallel
        parallel_map_helper(task_vec, threads, &all_group_sums);
    });

    // Combine group sums from all chunks
    std::vector<uint64_t> total_group_sums(num_groups + 1, 0);
    for (const auto &chunk_sums : all_group_sums) {
        for (size_t i = 0; i < chunk_sums.size() && i < total_group_sums.size(); i++) {
            total_group_sums[i] += chunk_sums[i];
        }
    }
    
    // Only keep the actual groups (exclude the last "discard" group)
    std::vector<uint64_t> final_group_sums(total_group_sums.begin(), total_group_sums.begin() + num_groups);

    // Read chunks - use version 9999 if bin_size == 1, otherwise use version 2
    std::vector<std::unique_ptr<MatrixLoader<uint32_t>>> matrix_chunks;
    for (size_t i = 0; i < chunks; i++) {
        FileReaderBuilder rb(chunk_output_paths[i]);
        if (bin_size == 1) {
            matrix_chunks.push_back(std::make_unique<StoredMatrix<uint32_t>>(EXPERIMENTAL_openPackedSparseColumn<uint32_t>(rb)));
        } else {
            matrix_chunks.push_back(std::make_unique<StoredMatrix<uint32_t>>(StoredMatrix<uint32_t>::openPacked(rb)));
        }
    }

    std::unique_ptr<MatrixLoader<uint32_t>> full_mat;
    if (chunks > 1) {
        full_mat = std::make_unique<ConcatCols<uint32_t>>(std::move(matrix_chunks), 0);
    } else {
        full_mat = std::move(matrix_chunks[0]);
    }
    
    // Unselect the last row, as those are the cells we meant to discard
    std::vector<uint32_t> row_selection;
    for (uint32_t i = 0; i < num_groups; i++) {
        row_selection.push_back(i);
    }
    full_mat = std::make_unique<MatrixRowSelect<uint32_t>>(std::move(full_mat), row_selection);
    
    // Set row names to group names using RenameDims
    std::vector<std::string> row_names;
    for (uint32_t i = 0; i < num_groups; i++) {
        row_names.push_back(actual_group_names[i]);
    }
    std::vector<std::string> empty_col_names;
    full_mat = std::make_unique<RenameDims<uint32_t>>(
        std::move(full_mat), row_names, empty_col_names, false, true
    );
    
    // Write final matrix - use version 9999 if bin_size == 1, otherwise use version 2
    FileWriterBuilder wb(output_path);
    
    if (bin_size == 1) {
        run_with_py_interrupt_check(
            &StoredMatrixWriter<uint32_t>::write,
            EXPERIMENTAL_createPackedSparseColumn<uint32_t>(wb),
            std::ref(*full_mat)
        );
    } else {
        run_with_py_interrupt_check(
            &StoredMatrixWriter<uint32_t>::write,
            StoredMatrixWriter<uint32_t>::createPacked(wb),
            std::ref(*full_mat)
        );
    }

    // Windows requires us to close open files before we can delete the temporary paths.
    full_mat.reset();

    for (const auto &x : chunk_output_paths) {
        std_fs::remove_all(std_fs::path(x));
    }
    
    // Write library sizes to JSON format
    if (!std_fs::exists(std_fs::path(output_path))) {
        std_fs::create_directories(std_fs::path(output_path));
    }
    
    std::string library_size_path = (std_fs::path(output_path) / "library_size.json").string();
    std::ofstream out_file(library_size_path);
    if (!out_file) {
        throw std::runtime_error("Could not open file for writing library sizes: " + library_size_path);
    }
    
    out_file << "{\n";
    out_file << "  \"library_sizes\": [\n";
    for (size_t i = 0; i < final_group_sums.size(); i++) {
        out_file << "    " << final_group_sums[i];
        if (i < final_group_sums.size() - 1) {
            out_file << ",";
        }
        out_file << "\n";
    }
    out_file << "  ]\n";
    out_file << "}\n";
    out_file.close();
}

void precalculate_pseudobulk_coverage_bam_multi(
    std::vector<std::string> bam_paths,
    std::vector<std::string> bam_prefixes,
    std::string output_path,
    std::string tmp_path,
    std::vector<std::string> chr,
    std::vector<uint32_t> chr_len,
    std::vector<int32_t> cell_groups,
    int shift_start,
    int shift_end,
    int bin_size,
    int threads,
    std::optional<std::vector<std::string>> group_names
) {
    // Validate inputs
    if (chr.size() != chr_len.size()) {
        throw std::runtime_error(
            "precalculate_pseudobulk_coverage_bam_multi: chr must be same length as chr_len"
        );
    }
    if (bam_paths.size() != bam_prefixes.size()) {
        throw std::runtime_error(
            "precalculate_pseudobulk_coverage_bam_multi: bam_paths and bam_prefixes must have same length"
        );
    }
    if (bam_paths.empty()) {
        throw std::runtime_error(
            "precalculate_pseudobulk_coverage_bam_multi: at least one BAM file required"
        );
    }

    // Create the arguments needed for TileMatrix: start, tile_width, chr_id, chr_levels
    std::vector<uint32_t> start(chr.size(), 0);
    std::vector<uint32_t> tile_width(chr.size(), bin_size);

    // Open first BAM to get chromosome names (all BAMs should have same chromosomes)
    BamFragments bam_frags(bam_paths[0]);
    std::vector<uint32_t> chr_id;
    std::unordered_map<std::string, uint32_t> chr_name_lookup;
    std::vector<std::string> chr_levels;
    
    int bam_chr_count = bam_frags.chrCount();
    for (int32_t i = 0; i < bam_chr_count; i++) {
        const char *chr_name = bam_frags.chrNames(i);
        if (chr_name == nullptr) {
            throw std::runtime_error("precalculate_pseudobulk_coverage_bam_multi: missing chr names in BAM");
        }
        chr_name_lookup[std::string(chr_name)] = i;
        chr_levels.push_back(std::string(chr_name));
    }
    
    for (auto &c : chr) {
        if (chr_name_lookup.find(c) == chr_name_lookup.end()) {
            throw std::runtime_error("precalculate_pseudobulk_coverage_bam_multi: chromosome " + c + " not found in BAM");
        }
        chr_id.push_back(chr_name_lookup[c]);
    }

    // Create the arguments needed for CellMerge (group_id, group_names)
    uint32_t num_groups = 1 + *std::max_element(cell_groups.cbegin(), cell_groups.cend());

    std::vector<std::string> actual_group_names;
    if (group_names.has_value()) {
        actual_group_names = group_names.value();
        // Make sure we have enough names, fill with defaults if needed
        if (actual_group_names.size() < num_groups) {
            for (size_t i = actual_group_names.size(); i < num_groups; i++) {
                actual_group_names.push_back(std::to_string(i));
            }
        }
        // Add one more for the discard group
        actual_group_names.push_back("discard");
    } else {
        // Generate dummy names if not provided
        for (uint32_t i = 0; i < num_groups + 1; i++) {
            actual_group_names.push_back(std::to_string(i));
        }
    }

    std::vector<uint32_t> cell_groups_uint;
    for (const auto &x : cell_groups) {
        cell_groups_uint.push_back(x >= 0 ? x : num_groups);
    }

    // Split columns into chunks
    size_t total_columns = 0;
    for (const auto &x : chr_len) {
        total_columns += (x + bin_size - 1) / bin_size;  // Ceiling division
    }
    uint32_t chunks = std::max<uint32_t>(1, threads * 4);
    std::vector<std::pair<uint32_t,uint32_t>> chunk_col_splits;
    uint32_t idx = 0;
    for (uint32_t i = 0; i < chunks; i++) {
        uint32_t col_count = (total_columns - idx) / (chunks - i);
        chunk_col_splits.push_back({idx, idx + col_count});
        idx += col_count;
    }

    std::vector<std::string> chunk_output_paths;
    for (uint32_t i = 0; i < chunks; i++) {
        chunk_output_paths.push_back((std_fs::path(tmp_path) / std::to_string(i)).string());
    }

    // Vector to store group sums from all chunks
    std::vector<std::vector<uint64_t>> all_group_sums;
    const std::vector<std::string> &group_names_ref = actual_group_names;

    // Make all the matrix chunks
    run_with_py_interrupt_check([&bam_paths,
                                        &bam_prefixes,
                                        &chunk_output_paths,
                                        &chunk_col_splits,
                                        &cell_groups_uint,
                                        &group_names_ref,
                                        &chr_id,
                                        &start,
                                        &chr_len,
                                        &tile_width,
                                        &chr_levels,
                                        &all_group_sums,
                                        threads,
                                        chunks,
                                        bin_size,
                                        shift_start,
                                        shift_end](std::atomic<bool> *user_interrupt) {
        std::vector<std::future<std::vector<uint64_t>>> task_vec;
        all_group_sums.resize(chunks);

        for (size_t i = 0; i < chunks; i++) {
            task_vec.push_back(std::async(
                std::launch::deferred,
                &precalculate_pseudobulk_coverage_bam_multi_helper,
                std::cref(bam_paths),
                std::cref(bam_prefixes),
                chunk_output_paths[i],
                chunk_col_splits[i],

                std::cref(cell_groups_uint),
                std::cref(group_names_ref),

                std::cref(chr_id),
                std::cref(start),
                std::cref(chr_len),
                std::cref(tile_width),
                std::cref(chr_levels),
                bin_size,
                shift_start,
                shift_end,

                user_interrupt
            ));
        }

        // Process the futures and collect the group sums in parallel
        parallel_map_helper(task_vec, threads, &all_group_sums);
    });

    // Combine group sums from all chunks
    std::vector<uint64_t> total_group_sums(num_groups + 1, 0);
    for (const auto &chunk_sums : all_group_sums) {
        for (size_t i = 0; i < chunk_sums.size() && i < total_group_sums.size(); i++) {
            total_group_sums[i] += chunk_sums[i];
        }
    }
    
    // Only keep the actual groups (exclude the last "discard" group)
    std::vector<uint64_t> final_group_sums(total_group_sums.begin(), total_group_sums.begin() + num_groups);

    // Read chunks - use version 9999 if bin_size == 1, otherwise use version 2
    std::vector<std::unique_ptr<MatrixLoader<uint32_t>>> matrix_chunks;
    for (size_t i = 0; i < chunks; i++) {
        FileReaderBuilder rb(chunk_output_paths[i]);
        if (bin_size == 1) {
            matrix_chunks.push_back(std::make_unique<StoredMatrix<uint32_t>>(EXPERIMENTAL_openPackedSparseColumn<uint32_t>(rb)));
        } else {
            matrix_chunks.push_back(std::make_unique<StoredMatrix<uint32_t>>(StoredMatrix<uint32_t>::openPacked(rb)));
        }
    }

    std::unique_ptr<MatrixLoader<uint32_t>> full_mat;
    if (chunks > 1) {
        full_mat = std::make_unique<ConcatCols<uint32_t>>(std::move(matrix_chunks), 0);
    } else {
        full_mat = std::move(matrix_chunks[0]);
    }
    
    // Unselect the last row, as those are the cells we meant to discard
    std::vector<uint32_t> row_selection;
    for (uint32_t i = 0; i < num_groups; i++) {
        row_selection.push_back(i);
    }
    full_mat = std::make_unique<MatrixRowSelect<uint32_t>>(std::move(full_mat), row_selection);
    
    // Set row names to group names using RenameDims
    std::vector<std::string> row_names;
    for (uint32_t i = 0; i < num_groups; i++) {
        row_names.push_back(actual_group_names[i]);
    }
    std::vector<std::string> empty_col_names;
    full_mat = std::make_unique<RenameDims<uint32_t>>(
        std::move(full_mat), row_names, empty_col_names, false, true
    );
    
    // Write final matrix - use version 9999 if bin_size == 1, otherwise use version 2
    FileWriterBuilder wb(output_path);
    
    if (bin_size == 1) {
        run_with_py_interrupt_check(
            &StoredMatrixWriter<uint32_t>::write,
            EXPERIMENTAL_createPackedSparseColumn<uint32_t>(wb),
            std::ref(*full_mat)
        );
    } else {
        run_with_py_interrupt_check(
            &StoredMatrixWriter<uint32_t>::write,
            StoredMatrixWriter<uint32_t>::createPacked(wb),
            std::ref(*full_mat)
        );
    }

    // Windows requires us to close open files before we can delete the temporary paths.
    full_mat.reset();

    for (const auto &x : chunk_output_paths) {
        std_fs::remove_all(std_fs::path(x));
    }
    
    // Write library sizes to JSON format (as dict mapping group names to sizes)
    if (!std_fs::exists(std_fs::path(output_path))) {
        std_fs::create_directories(std_fs::path(output_path));
    }
    
    std::string library_size_path = (std_fs::path(output_path) / "library_size.json").string();
    std::ofstream out_file(library_size_path);
    if (!out_file) {
        throw std::runtime_error("Could not open file for writing library sizes: " + library_size_path);
    }
    
    // Write as dict mapping group names to library sizes
    out_file << "{\n";
    for (size_t i = 0; i < final_group_sums.size(); i++) {
        std::string group_name = (i < actual_group_names.size()) ? actual_group_names[i] : std::to_string(i);
        out_file << "  \"" << group_name << "\": " << final_group_sums[i];
        if (i < final_group_sums.size() - 1) {
            out_file << ",";
        }
        out_file << "\n";
    }
    out_file << "}\n";
    out_file.close();
}

} // namespace BPCells::py

