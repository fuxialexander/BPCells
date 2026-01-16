// Copyright 2023 BPCells contributors
// 
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>

#include "py_interrupts.hpp"
#include "fragments.hpp"
#include "bam.hpp"
#include "matrix.hpp"

#include "bpcells-cpp/arrayIO/vector.h"

#include "bpcells-cpp/simd/bp128.h"
#include "bpcells-cpp/simd/current_target.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

using namespace BPCells;


PYBIND11_MODULE(cpp, m) {

    m.def("import_10x_fragments", &BPCells::py::import_10x_fragments);
    m.def("cell_names_fragments_dir", &BPCells::py::cell_names_fragments_dir);
    m.def("chr_names_fragments_dir", &BPCells::py::chr_names_fragments_dir);
    m.def("pseudobulk_coverage", &BPCells::py::pseudobulk_coverage);
    m.def("precalculate_pseudobulk_coverage", &BPCells::py::precalculate_pseudobulk_coverage,
          pybind11::arg("fragments_path"), pybind11::arg("output_path"),
          pybind11::arg("tmp_path"), pybind11::arg("chr"), pybind11::arg("chr_size"),
          pybind11::arg("cell_groups"), pybind11::arg("bin_size"), pybind11::arg("threads"),
          pybind11::arg("group_names") = nullptr,
          pybind11::arg("preserve_chrom_order") = false);
           m.def("bam_has_cb_tags", &BPCells::py::bam_has_cb_tags,
                 pybind11::arg("bam_path"), pybind11::arg("barcode_tag") = "CB", pybind11::arg("sample_size") = 10000);
           m.def("discover_cells_from_bam", &BPCells::py::discover_cells_from_bam,
                 pybind11::arg("bam_path"), pybind11::arg("barcode_tag") = "CB", pybind11::arg("cell_prefix") = "");
           m.def("discover_cells_from_bam_multi", &BPCells::py::discover_cells_from_bam_multi,
                 pybind11::arg("bam_paths"), pybind11::arg("bam_prefixes"), pybind11::arg("barcode_tag") = "CB");
           m.def("precalculate_pseudobulk_coverage_bam", &BPCells::py::precalculate_pseudobulk_coverage_bam,
                 pybind11::arg("bam_path"), pybind11::arg("output_path"),
                 pybind11::arg("tmp_path"), pybind11::arg("chr"), pybind11::arg("chr_len"),
                 pybind11::arg("cell_groups"), pybind11::arg("shift_start"), pybind11::arg("shift_end"),
                 pybind11::arg("bin_size"), pybind11::arg("threads"),
                 pybind11::arg("group_names") = pybind11::none(),
                 pybind11::arg("signal_mode") = 0);  // 0 = tn5 (insertion), 1 = chip (coverage)
           m.def("precalculate_pseudobulk_coverage_bam_multi", &BPCells::py::precalculate_pseudobulk_coverage_bam_multi,
                 pybind11::arg("bam_paths"), pybind11::arg("bam_prefixes"), pybind11::arg("output_path"),
                 pybind11::arg("tmp_path"), pybind11::arg("chr"), pybind11::arg("chr_len"),
                 pybind11::arg("cell_groups"), pybind11::arg("shift_start"), pybind11::arg("shift_end"),
                 pybind11::arg("bin_size"), pybind11::arg("threads"),
                 pybind11::arg("group_names") = pybind11::none(),
                 pybind11::arg("signal_mode") = 0);  // 0 = tn5 (insertion), 1 = chip (coverage)
    m.def("query_precalculated_pseudobulk_coverage", &BPCells::py::query_precalculated_pseudobulk_coverage);
        
    m.def("write_matrix_dir_from_memory", &BPCells::py::write_matrix_dir_from_memory);
    m.def("write_matrix_dir_from_memory_experimental", &BPCells::py::write_matrix_dir_from_memory_experimental);
    m.def("write_matrix_dir_from_concat", &BPCells::py::write_matrix_dir_from_concat);
    m.def("write_matrix_dir_from_concat_experimental", &BPCells::py::write_matrix_dir_from_concat_experimental,
          pybind11::arg("in_paths"),
          pybind11::arg("out_path"),
          pybind11::arg("concat_rows"),
          pybind11::arg("batch_size") = 16,
          pybind11::arg("threads") = 1,
          pybind11::arg("temp_dir") = "");
    m.def("write_matrix_dir_from_h5ad", &BPCells::py::write_matrix_dir_from_h5ad);
    
    m.def("load_matrix_dir_subset", &BPCells::py::load_matrix_dir_subset);
    m.def("dims_matrix_dir", &BPCells::py::dims_matrix_dir);
    m.def("row_names_stored_matrix", &BPCells::py::row_names_stored_matrix);

    m.def("load_matrix_dir_to_memory", &BPCells::py::load_matrix_dir_to_memory);
    m.def("load_matrix_memory_subset", &BPCells::py::load_matrix_memory_subset);
    pybind11::class_<VecReaderWriterBuilder, std::shared_ptr<VecReaderWriterBuilder>>(m, "VecReaderWriterBuilder");
    
    m.def("simd_current_target", &BPCells::simd::current_target);
    m.def("simd_current_target_bp128", &BPCells::simd::bp128::current_target);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

}
