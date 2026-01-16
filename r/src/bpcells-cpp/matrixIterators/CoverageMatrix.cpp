// Copyright 2024 BPCells contributors
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#include "CoverageMatrix.h"

#include "../simd/math.h"
#include "../simd/overlaps.h"

namespace BPCells {

CoverageMatrix::CoverageMatrix(
    std::unique_ptr<FragmentLoader> &&frags,
    const std::vector<uint32_t> &chr,
    const std::vector<uint32_t> &start,
    const std::vector<uint32_t> &end,
    const std::vector<uint32_t> &width,
    std::unique_ptr<StringReader> &&chr_levels
)
    : frags(std::move(frags))
    , chr_levels(std::move(chr_levels)) {
    if (this->frags->cellCount() < 0)
        throw std::invalid_argument(
            "frags must have a known cell count. Consider using a cell selection to define the "
            "number of cells."
        );

    if (chr.size() != start.size() || chr.size() != end.size() || chr.size() != width.size())
        throw std::invalid_argument("chr, start, end, and width must all be same length");

    // Check that chr name matches
    for (uint32_t i = 0; i < this->chr_levels->size(); i++) {
        const char *chr_name_frag = this->frags->chrNames(i);
        const char *chr_name_args = this->chr_levels->get(i);
        if (chr_name_frag != NULL &&
            (chr_name_args == NULL || strcmp(chr_name_frag, chr_name_args) != 0)) {
            throw std::runtime_error(
                std::string("CoverageMatrix encountered fragment with incorrect chrLevel: ") +
                std::string(chr_name_frag) + std::string(" expected: ") + std::string(chr_name_args)
            );
        }
    }

    Tile prev;
    for (size_t i = 0; i < chr.size(); i++) {
        if (chr[i] >= this->chr_levels->size())
            throw std::invalid_argument("chr has values higher than length of chr_levels");
        if (width[i] == 0)
            throw std::invalid_argument("0 widths not supported");

        Tile t;
        t.start = start[i];
        t.end = end[i];
        t.chr = chr[i];
        t.width = width[i];
        t.output_idx = n_tiles;
        sorted_tiles.push_back(t);

        n_tiles += (end[i] - start[i] + width[i] - 1) / width[i];
        if (i > 0) {
            bool ordered = true;
            if (prev.chr != t.chr) ordered = prev.chr < t.chr;
            else if (prev.end > t.start) ordered = false;
            if (!ordered) {
                throw std::invalid_argument(
                    "Tiles are not sorted by (chr,start) and non-overlapping"
                );
            }
        }
        prev = t;
    }

    // Sentinel value
    sorted_tiles.push_back({UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, 1});

    // Initialize
    if (!this->frags->nextChr()) {
        next_completed_tile = UINT32_MAX;
        active_tiles.clear();
        return;
    }

    const char *chr_name_frag = this->frags->chrNames(this->frags->currentChr());
    const char *chr_name_args = this->chr_levels->get(this->frags->currentChr());
    if (chr_name_frag == NULL || chr_name_args == NULL ||
        strcmp(chr_name_frag, chr_name_args) != 0) {
        throw std::runtime_error(
            std::string("CoverageMatrix encountered fragment with incorrect chrLevel: ") +
            std::string(chr_name_frag) + std::string(" expected: ") + std::string(chr_name_args)
        );
    }
    while (sorted_tiles[next_active_tile].chr < this->frags->currentChr()) {
        next_active_tile++;
    }
    next_completed_tile = sorted_tiles[next_active_tile].output_idx;
}

uint32_t CoverageMatrix::rows() const { return frags->cellCount(); }
uint32_t CoverageMatrix::cols() const { return n_tiles; }

const char *CoverageMatrix::rowNames(uint32_t row) { return frags->cellNames(row); }
const char *CoverageMatrix::colNames(uint32_t col) {
    if (col >= cols()) return NULL;
    auto tile = std::upper_bound(
                    sorted_tiles.begin(),
                    sorted_tiles.end(),
                    col,
                    [](uint32_t col, Tile t) { return col < t.output_idx; }
                ) -
                1;

    uint32_t start_base = tile->start + tile->width * (col - tile->output_idx);

    tile_name.clear();
    tile_name += frags->chrNames(tile->chr);
    tile_name += ":";
    tile_name += std::to_string(start_base);
    tile_name += "-";
    tile_name += std::to_string(std::min(tile->end, start_base + tile->width));
    return tile_name.c_str();
}

void CoverageMatrix::restart() {
    accumulator.clear();
    active_tiles.clear();
    next_completed_tile = 0;
    current_output_tile = UINT32_MAX;
    next_active_tile = 0;
}

void CoverageMatrix::seekCol(uint32_t col) {
    if (!frags->isSeekable())
        throw std::runtime_error("Can't seek a CoverageMatrix if the fragments aren't seekable");

    auto next_tile =
        std::upper_bound(sorted_tiles.begin(), sorted_tiles.end(), col, [](uint32_t value, Tile t) {
            return value < t.output_idx;
        });

    next_active_tile = (next_tile - sorted_tiles.begin()) - 1;
    next_completed_tile = 0;
    current_output_tile = col - 1;
    active_tiles.clear();
    accumulator.clear();
    nextCol();
}

bool CoverageMatrix::nextCol() {
    current_output_tile += 1;

    if (current_output_tile >= cols()) {
        current_output_tile -= 1;
        return false;
    }
    if (current_output_tile >= next_completed_tile) loadFragments();
    accumulator.discard_until(current_output_tile);
    return true;
}

uint32_t CoverageMatrix::currentCol() const { return current_output_tile; }

bool CoverageMatrix::load() { return accumulator.load(current_output_tile, 1024); }

uint32_t CoverageMatrix::capacity() const { return accumulator.capacity(); }

uint32_t *CoverageMatrix::rowData() { return accumulator.rowData(); }
uint32_t *CoverageMatrix::valData() { return accumulator.valData(); }

void CoverageMatrix::loadFragments() {
    if (next_active_tile == sorted_tiles.size()) return;

    if (active_tiles.size() == 0 && frags->isSeekable()) {
        uint32_t seek_bp = sorted_tiles[next_active_tile].start;
        if (current_output_tile > sorted_tiles[next_active_tile].output_idx &&
            current_output_tile < sorted_tiles[next_active_tile + 1].output_idx) {
            seek_bp = sorted_tiles[next_active_tile].start +
                      (current_output_tile - sorted_tiles[next_active_tile].output_idx) *
                          sorted_tiles[next_active_tile].width;
        }
        frags->seek(sorted_tiles[next_active_tile].chr, seek_bp);
    }

    while (true) {
        while (!frags->load()) {
            uint32_t prev_chr_id = frags->currentChr();

            if (!frags->nextChr()) {
                next_completed_tile = UINT32_MAX;
                active_tiles.clear();
                return;
            }
            if (frags->currentChr() <= prev_chr_id) {
                throw std::runtime_error(
                    "CoverageMatrix encountered fragments with out of order chromosome IDs."
                );
            }
            const char *chr_name_frag = frags->chrNames(frags->currentChr());
            const char *chr_name_args = chr_levels->get(frags->currentChr());
            if (chr_name_frag == NULL || chr_name_args == NULL ||
                strcmp(chr_name_frag, chr_name_args) != 0) {
                throw std::runtime_error(
                    std::string("CoverageMatrix encountered fragment with incorrect chrLevel: ") +
                    std::string(chr_name_frag) + std::string(" expected: ") +
                    std::string(chr_name_args)
                );
            }
            while (sorted_tiles[next_active_tile].chr < frags->currentChr()) {
                next_active_tile++;
            }
            next_completed_tile = sorted_tiles[next_active_tile].output_idx;
            active_tiles.clear();
        }

        uint32_t capacity = frags->capacity();
        uint32_t *start_data = frags->startData();
        uint32_t *end_data = frags->endData();
        uint32_t *cell_data = frags->cellData();

        uint32_t i = 0;
        uint32_t end_max = 0;

        // Buffer for coverage results (need larger buffer since fragments can span tiles)
        const uint32_t max_items = 256;
        const uint32_t max_output = max_items * 100; // Assume max 100 tiles per fragment
        uint32_t overlap_cell[max_output];
        uint32_t overlap_tile[max_output];
        uint32_t overlap_count[max_output];

        while (i < capacity) {
            uint32_t items = std::min(max_items, capacity - i);
            end_max = std::max(end_max, simd::max(end_data + i, items));

            // Check for new tiles to activate
            while (sorted_tiles[next_active_tile].chr == frags->currentChr() &&
                   sorted_tiles[next_active_tile].start < end_max) {
                active_tiles.push_back(sorted_tiles[next_active_tile]);
                next_active_tile += 1;
            }

            // For each active tile, calculate coverage
            for (uint32_t j = 0; j < active_tiles.size(); j++) {
                const Tile t = active_tiles[j];

                uint32_t n_overlaps = simd::tile_overlaps_coverage(
                    cell_data + i,
                    start_data + i,
                    end_data + i,
                    items,
                    t.start,
                    t.end,
                    t.output_idx,
                    t.width,
                    overlap_cell,
                    overlap_tile,
                    overlap_count
                );

                for (uint32_t k = 0; k < n_overlaps; k++) {
                    accumulator.add_one(overlap_tile[k], overlap_cell[k], overlap_count[k]);
                }

                // Remove the tile if we're done
                if (start_data[i + items - 1] >= t.end) {
                    std::swap(active_tiles.back(), active_tiles[j]);
                    active_tiles.pop_back();
                    j -= 1;
                }
            }
            i += items;
        }

        // Update next_completed_tile
        if (capacity > 0) {
            auto max_region = std::upper_bound(
                sorted_tiles.begin(),
                sorted_tiles.end(),
                std::pair{frags->currentChr(), start_data[capacity - 1]},
                [](std::pair<uint32_t, uint32_t> value, Tile t) {
                    if (value.first != t.chr) return value.first < t.chr;
                    return value.second < t.start;
                }
            );

            if (max_region != sorted_tiles.begin()) {
                uint32_t new_completed_tile;
                Tile t = *(max_region - 1);
                if (t.chr == frags->currentChr() && t.start <= start_data[capacity - 1] &&
                    t.end > start_data[capacity - 1]) {
                    new_completed_tile =
                        t.output_idx + (start_data[capacity - 1] - t.start) / t.width;
                } else {
                    new_completed_tile = max_region->output_idx;
                }
                next_completed_tile = std::max(next_completed_tile, new_completed_tile);
            }
        }

        if (accumulator.ready_for_loading() && next_completed_tile > current_output_tile) {
            break;
        }
    }
}

} // end namespace BPCells
