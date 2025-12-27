// Copyright 2023 BPCells contributors
// 
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#pragma once

#include <optional>
#include <string>
#include <vector>

namespace BPCells::py {

// Calculate pseudobulk coverage directly from BAM file
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
    std::optional<std::vector<std::string>> group_names = std::nullopt
);

// Calculate pseudobulk coverage directly from multiple BAM files
// Each BAM file gets a unique cell prefix (e.g., "bulk.FILENAME") to distinguish cells
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
    std::optional<std::vector<std::string>> group_names = std::nullopt
);

} // namespace BPCells::py

