// Copyright 2024 BPCells contributors
// 
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#pragma once

#include "FragmentIterator.h"
#include <htslib/sam.h>
#include <string>
#include <vector>
#include <unordered_map>

namespace BPCells {

// Read fragments directly from a BAM file
// Extracts proper pairs and uses CB tag for cell barcode
class BamFragments : public FragmentLoader {
private:
    samFile *fp = nullptr;
    bam_hdr_t *header = nullptr;
    hts_idx_t *idx = nullptr;
    hts_itr_t *iter = nullptr;
    bam1_t *b = nullptr;

    std::string path;
    std::string barcode_tag;
    std::string cell_prefix;

    std::vector<uint32_t> cell_buf, start_buf, end_buf;
    uint32_t current_chr_id = UINT32_MAX;
    bool done_with_chr = false;
    bool eof = false;

    std::vector<std::string> chr_names;
    std::unordered_map<std::string, uint32_t> chr_lookup;
    std::vector<std::string> cell_names;
    std::unordered_map<std::string, uint32_t> cell_id_lookup;
    uint32_t next_cell_id = 0;

    // Buffer size for loading fragments
    static constexpr uint32_t BUFFER_SIZE = 1024;

    // Helper to get cell ID from barcode tag
    uint32_t getCellId(const bam1_t *bam);

    // Helper to check if read is a valid fragment (proper pair, etc.)
    bool isValidFragment(const bam1_t *bam);

public:
    // Constructor opens file and loads index
    // barcode_tag: BAM tag to use for cell barcode (default "CB")
    // cell_prefix: Optional prefix to add to cell barcodes
    BamFragments(std::string path, std::string barcode_tag = "CB", std::string cell_prefix = "");
    
    ~BamFragments();

    BamFragments() = delete;
    BamFragments(const BamFragments &) = delete;
    BamFragments &operator=(const BamFragments &other) = delete;

    bool isSeekable() const override { return true; }
    void seek(uint32_t chr_id, uint32_t base) override;
    void restart() override;

    int chrCount() const override;
    int cellCount() const override { return cell_names.size(); } // Return discovered cell count

    const char *chrNames(uint32_t chr_id) override;
    const char *cellNames(uint32_t cell_id) override;

    bool nextChr() override;
    uint32_t currentChr() const override { return current_chr_id; }

    bool load() override; // Main parsing logic here
    uint32_t capacity() const override { return cell_buf.size(); }

    uint32_t *cellData() override { return cell_buf.data(); }
    uint32_t *startData() override { return start_buf.data(); }
    uint32_t *endData() override { return end_buf.data(); }
};

} // end namespace BPCells

