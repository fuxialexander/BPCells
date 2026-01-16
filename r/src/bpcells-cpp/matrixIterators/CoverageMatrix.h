// Copyright 2024 BPCells contributors
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#pragma once

#include <algorithm>

#include "../arrayIO/array_interfaces.h"
#include "../fragmentIterators/FragmentIterator.h"
#include "MatrixAccumulators.h"
#include "MatrixIterator.h"

namespace BPCells {

// Output cell x tile matrix with coverage counts (rows = cell_id, col = tile_id)
// For ChIP-seq data: counts the number of base pairs covered by each cell in each tile
// Unlike TileMatrix which counts insertion sites, this counts full fragment coverage
class CoverageMatrix : public MatrixLoader<uint32_t> {
  private:
    class Tile {
      public:
        uint32_t chr, start, end, output_idx, width;
    };

    std::unique_ptr<FragmentLoader> frags;
    std::unique_ptr<StringReader> chr_levels;
    MatrixAccumulator<uint32_t> accumulator;
    std::vector<Tile> sorted_tiles;
    std::vector<Tile> active_tiles;
    uint32_t next_completed_tile = 0;
    uint32_t current_output_tile = UINT32_MAX;
    uint32_t next_active_tile = 0;
    uint32_t n_tiles = 0;

    std::string tile_name;

    void loadFragments();

  public:
    CoverageMatrix(
        std::unique_ptr<FragmentLoader> &&frags,
        const std::vector<uint32_t> &chr,
        const std::vector<uint32_t> &start,
        const std::vector<uint32_t> &end,
        const std::vector<uint32_t> &width,
        std::unique_ptr<StringReader> &&chr_levels
    );

    uint32_t rows() const override;
    uint32_t cols() const override;

    const char *rowNames(uint32_t row) override;
    const char *colNames(uint32_t col) override;

    void restart() override;
    void seekCol(uint32_t col) override;

    bool nextCol() override;

    uint32_t currentCol() const override;

    bool load() override;

    uint32_t capacity() const override;

    uint32_t *rowData() override;
    uint32_t *valData() override;
};

} // end namespace BPCells
