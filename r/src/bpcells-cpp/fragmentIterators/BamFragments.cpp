// Copyright 2024 BPCells contributors
// 
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#include "BamFragments.h"
#include <stdexcept>
#include <cstring>

namespace BPCells {

BamFragments::BamFragments(std::string path, std::string barcode_tag, std::string cell_prefix)
    : path(path)
    , barcode_tag(barcode_tag)
    , cell_prefix(cell_prefix) {
    restart();
}

BamFragments::~BamFragments() {
    if (iter != nullptr) {
        hts_itr_destroy(iter);
        iter = nullptr;
    }
    if (b != nullptr) {
        bam_destroy1(b);
        b = nullptr;
    }
    if (idx != nullptr) {
        hts_idx_destroy(idx);
        idx = nullptr;
    }
    if (header != nullptr) {
        bam_hdr_destroy(header);
        header = nullptr;
    }
    if (fp != nullptr) {
        sam_close(fp);
        fp = nullptr;
    }
}

void BamFragments::restart() {
    // Clean up existing resources
    if (iter != nullptr) {
        hts_itr_destroy(iter);
        iter = nullptr;
    }
    if (b != nullptr) {
        bam_destroy1(b);
        b = nullptr;
    }
    if (idx != nullptr) {
        hts_idx_destroy(idx);
        idx = nullptr;
    }
    if (header != nullptr) {
        bam_hdr_destroy(header);
        header = nullptr;
    }
    if (fp != nullptr) {
        sam_close(fp);
        fp = nullptr;
    }

    // Reset state
    current_chr_id = UINT32_MAX;
    done_with_chr = false;
    eof = false;
    chr_names.clear();
    chr_lookup.clear();
    cell_names.clear();
    cell_id_lookup.clear();
    next_cell_id = 0;
    cell_buf.clear();
    start_buf.clear();
    end_buf.clear();

    // Open BAM file
    fp = sam_open(path.c_str(), "r");
    if (fp == nullptr) {
        throw std::runtime_error("Failed to open BAM file: " + path);
    }

    // Load header
    header = sam_hdr_read(fp);
    if (header == nullptr) {
        sam_close(fp);
        fp = nullptr;
        throw std::runtime_error("Failed to read BAM header: " + path);
    }

    // Build chromosome lookup from header
    for (int i = 0; i < header->n_targets; i++) {
        std::string chr_name(header->target_name[i]);
        chr_lookup[chr_name] = chr_names.size();
        chr_names.push_back(chr_name);
    }

    // Load index
    idx = sam_index_load(fp, path.c_str());
    if (idx == nullptr) {
        // Index not found - this is OK, we just can't seek
        // But we'll still be able to read sequentially
    }

    // Allocate bam1_t structure
    b = bam_init1();
    if (b == nullptr) {
        throw std::runtime_error("Failed to allocate bam1_t structure");
    }

    // Start at first chromosome
    if (chr_names.size() > 0) {
        current_chr_id = 0;
        done_with_chr = false;
    } else {
        eof = true;
    }
    
    // Pre-initialize "bulk" cell for bulk data (when no CB tags are present)
    // This ensures cellCount() returns at least 1 even before any fragments are loaded
    std::string bulk_cell = cell_prefix + "bulk";
    cell_id_lookup[bulk_cell] = 0;
    cell_names.push_back(bulk_cell);
    next_cell_id = 1;
}

bool BamFragments::isValidFragment(const bam1_t *bam) {
    // Check if it's a proper pair (both reads mapped, correct orientation)
    if (!(bam->core.flag & BAM_FPROPER_PAIR)) {
        return false;
    }
    
    // Check if both reads are mapped
    if (bam->core.flag & BAM_FUNMAP || bam->core.flag & BAM_FMUNMAP) {
        return false;
    }
    
    // Only process first read in pair to avoid duplicates
    if (!(bam->core.flag & BAM_FREAD1)) {
        return false;
    }
    
    // Check mapping quality (optional - you might want to filter low quality)
    // if (bam->core.qual < 10) return false;
    
    return true;
}

uint32_t BamFragments::getCellId(const bam1_t *bam) {
    uint8_t *tag_data = bam_aux_get(bam, barcode_tag.c_str());
    if (tag_data == nullptr) {
        // No barcode tag - use "bulk" as dummy barcode for bulk data
        std::string bulk_cell = cell_prefix + "bulk";
        auto cell_id_res = cell_id_lookup.emplace(bulk_cell, next_cell_id);
        if (cell_id_res.second) {
            cell_names.push_back(bulk_cell);
            next_cell_id++;
        }
        return cell_id_res.first->second;
    }

    // Extract barcode string (assuming it's a Z-type tag)
    if (tag_data[0] == 'Z') {
        const char *barcode = (const char *)(tag_data + 1);
        std::string full_barcode = cell_prefix + std::string(barcode);
        auto cell_id_res = cell_id_lookup.emplace(full_barcode, next_cell_id);
        if (cell_id_res.second) {
            cell_names.push_back(full_barcode);
            next_cell_id++;
        }
        return cell_id_res.first->second;
    }

    // If tag is not Z-type, treat as bulk
    std::string bulk_cell = cell_prefix + "bulk";
    auto cell_id_res = cell_id_lookup.emplace(bulk_cell, next_cell_id);
    if (cell_id_res.second) {
        cell_names.push_back(bulk_cell);
        next_cell_id++;
    }
    return cell_id_res.first->second;
}

int BamFragments::chrCount() const {
    return chr_names.size();
}

const char *BamFragments::chrNames(uint32_t chr_id) {
    if (chr_id >= chr_names.size()) return nullptr;
    return chr_names[chr_id].c_str();
}

const char *BamFragments::cellNames(uint32_t cell_id) {
    if (cell_id >= cell_names.size()) return nullptr;
    return cell_names[cell_id].c_str();
}

bool BamFragments::nextChr() {
    if (eof || current_chr_id == UINT32_MAX) return false;
    
    // Destroy current iterator if it exists
    if (iter != nullptr) {
        hts_itr_destroy(iter);
        iter = nullptr;
    }
    
    // Move to next chromosome
    current_chr_id++;
    if (current_chr_id >= chr_names.size()) {
        eof = true;
        return false;
    }
    
    done_with_chr = false;
    return true;
}

void BamFragments::seek(uint32_t chr_id, uint32_t base) {
    if (idx == nullptr) {
        throw std::runtime_error("Cannot seek: BAM index not available");
    }
    
    if (chr_id >= chr_names.size()) {
        throw std::runtime_error("Invalid chromosome ID for seek");
    }
    
    // Destroy existing iterator
    if (iter != nullptr) {
        hts_itr_destroy(iter);
        iter = nullptr;
    }
    
    // Create iterator for the chromosome starting at base position
    // BAM coordinates are 0-based, same as BPCells
    iter = sam_itr_queryi(idx, chr_id, base, header->target_len[chr_id]);
    if (iter == nullptr) {
        // No reads in this region, but that's OK
        done_with_chr = true;
        return;
    }
    
    current_chr_id = chr_id;
    done_with_chr = false;
}

bool BamFragments::load() {
    if (eof || current_chr_id == UINT32_MAX) return false;
    if (done_with_chr) return false;
    
    // Clear buffers
    cell_buf.clear();
    start_buf.clear();
    end_buf.clear();
    cell_buf.reserve(BUFFER_SIZE);
    start_buf.reserve(BUFFER_SIZE);
    end_buf.reserve(BUFFER_SIZE);
    
    int ret;
    uint32_t last_start = 0;
    bool first_read = true;
    
    // If we have an iterator (from seek), use it; otherwise read sequentially
    if (iter != nullptr) {
        // Read from iterator
        while ((ret = sam_itr_next(fp, iter, b)) >= 0) {
            if (!isValidFragment(b)) continue;
            
            // Check chromosome matches
            if (b->core.tid != (int)current_chr_id) {
                // We've moved to a different chromosome
                done_with_chr = true;
                break;
            }
            
            // Extract fragment coordinates
            // BAM is 0-based, BPCells expects 0-based half-open
            // Start is already 0-based
            uint32_t frag_start = b->core.pos;
            // End is calculated from template length (isize/TLEN) for proper pairs
            // isize is signed: positive means read1 is leftmost, negative means read2 is leftmost
            // abs(isize) is the insert size (distance from leftmost to rightmost mapped base)
            uint32_t frag_end;
            if (b->core.tid == b->core.mtid && b->core.flag & BAM_FPROPER_PAIR && b->core.isize != 0) {
                // Proper pair on same chromosome with valid template length
                // Template length gives distance from leftmost to rightmost base
                // For half-open coordinates, we want one past the rightmost base
                frag_end = frag_start + abs(b->core.isize);
            } else {
                // Fallback: use read end position (one past last base)
                frag_end = bam_endpos(b);
            }
            
            // Ensure end > start
            if (frag_end <= frag_start) continue;
            
            // Check sorting (fragments should be sorted by start position)
            if (!first_read && frag_start < last_start) {
                // Not sorted - but continue anyway
            }
            last_start = frag_start;
            first_read = false;
            
            // Extract cell ID
            uint32_t cell_id = getCellId(b);
            
            // Add to buffers
            cell_buf.push_back(cell_id);
            start_buf.push_back(frag_start);
            end_buf.push_back(frag_end);
            
            if (cell_buf.size() >= BUFFER_SIZE) break;
        }
        
        if (ret < -1) {
            // Error reading
            throw std::runtime_error("Error reading from BAM iterator");
        }
        
        if (ret < 0) {
            // End of iterator
            done_with_chr = true;
        }
    } else {
        // Sequential reading - read until we hit a different chromosome or EOF
        while ((ret = sam_read1(fp, header, b)) >= 0) {
            if (!isValidFragment(b)) continue;
            
            // Check if we've moved to a different chromosome
            if (b->core.tid < 0 || (uint32_t)b->core.tid != current_chr_id) {
                // We've finished this chromosome
                done_with_chr = true;
                break;
            }
            
            // Extract fragment coordinates
            uint32_t frag_start = b->core.pos;
            uint32_t frag_end;
            if (b->core.tid == b->core.mtid && b->core.flag & BAM_FPROPER_PAIR && b->core.isize != 0) {
                // Proper pair on same chromosome with valid template length
                frag_end = frag_start + abs(b->core.isize);
            } else {
                // Fallback: use read end position
                frag_end = bam_endpos(b);
            }
            
            if (frag_end <= frag_start) continue;
            
            if (!first_read && frag_start < last_start) {
                // Not sorted
            }
            last_start = frag_start;
            first_read = false;
            
            uint32_t cell_id = getCellId(b);
            
            cell_buf.push_back(cell_id);
            start_buf.push_back(frag_start);
            end_buf.push_back(frag_end);
            
            if (cell_buf.size() >= BUFFER_SIZE) break;
        }
        
        if (ret < -1) {
            throw std::runtime_error("Error reading from BAM file");
        }
        
        if (ret < 0) {
            // EOF
            eof = true;
            done_with_chr = true;
        }
    }
    
    return cell_buf.size() > 0;
}

} // end namespace BPCells

