// Copyright 2023 BPCells contributors
// 
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace BPCells::py {

// Quick check if BAM file has CB tags (samples first N reads)
// Returns true if CB tags found, false otherwise
bool bam_has_cb_tags(std::string bam_path, std::string barcode_tag = "CB", uint32_t sample_size = 10000);

// Discover all cells from BAM file (full scan)
// Returns vector of cell names in discovery order
std::vector<std::string> discover_cells_from_bam(
    std::string bam_path,
    std::string barcode_tag = "CB",
    std::string cell_prefix = ""
);

// Discover all cells from multiple BAM files
std::vector<std::string> discover_cells_from_bam_multi(
    std::vector<std::string> bam_paths,
    std::vector<std::string> bam_prefixes,
    std::string barcode_tag = "CB"
);

// Calculate pseudobulk coverage directly from BAM file
// signal_mode: 0 = tn5 (insertion counts for ATAC-seq), 1 = chip (coverage counts for ChIP-seq)
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
    std::optional<std::vector<std::string>> group_names = std::nullopt,
    int signal_mode = 0  // 0 = tn5 (insertion), 1 = chip (coverage)
);

// Calculate pseudobulk coverage directly from multiple BAM files
// Each BAM file gets a unique cell prefix (e.g., "bulk.FILENAME") to distinguish cells
// signal_mode: 0 = tn5 (insertion counts for ATAC-seq), 1 = chip (coverage counts for ChIP-seq)
void precalculate_pseudobulk_coverage_bam_multi(
    std::vector<std::string> bam_paths,
    std::vector<std::string> bam_prefixes,  // Cell prefixes for each BAM (e.g., "bulk.FILENAME")
    std::string output_path,
    std::string tmp_path,
    std::vector<std::string> chr,
    std::vector<uint32_t> chr_len,
    std::vector<int32_t> cell_groups,
    int shift_start,
    int shift_end,
    int bin_size,
    int threads,
    std::optional<std::vector<std::string>> group_names = std::nullopt,
    int signal_mode = 0  // 0 = tn5 (insertion), 1 = chip (coverage)
);

} // namespace BPCells::py

