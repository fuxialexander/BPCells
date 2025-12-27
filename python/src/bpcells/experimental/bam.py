# Copyright 2024 BPCells contributors
# 
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

"""BAM file processing functions for BPCells.

This module provides functions to process BAM files directly without requiring
conversion to BPCells fragment format first.
"""

import bpcells
import bpcells.cpp

import json
import os.path
import shutil
import tempfile
import warnings

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .fragments import PrecalculatedInsertionMatrix


def precalculate_insertion_counts_bam(
    bam_file: str,
    output_dir: str,
    cell_groups: Dict[str, int],
    chrom_sizes: Union[str, Dict[str, int]],
    shift_start: int = 4,
    shift_end: int = -5,
    threads: int = 0,
    group_names: Optional[Union[List[str], Dict[str, str]]] = None
) -> PrecalculatedInsertionMatrix:
    """Precalculate per-base insertion counts directly from BAM file.

    This function reads fragments directly from a BAM file without requiring
    conversion to BPCells fragment format first. It supports both single-cell
    ATAC-seq data (with CB tags) and bulk ATAC-seq data (without CB tags).

    The function extracts proper paired-end reads, applies Tn5 shift correction,
    and calculates per-base insertion counts for pseudobulk groups. The output
    is compatible with :class:`CelltypeDenseBPCellsIO` for downstream analysis.

    **Fragment Processing:**
    - Only proper pairs (both reads mapped, correct orientation) are processed
    - Only read1 is used to avoid duplicate counting
    - Fragment coordinates are extracted from template length (isize/TLEN)
    - Tn5 shift is applied: start +4bp, end -5bp (default for ATAC-seq)

    **Cell Barcode Handling:**
    - Single-cell data: Uses CB tag to identify cells
    - Bulk data: Automatically assigns a "bulk" dummy barcode if no CB tags found
    - Cells are discovered dynamically as the BAM is read

    **Group Assignment:**
    - Each cell barcode is mapped to a group ID via the `cell_groups` dictionary
    - Cells not in `cell_groups` are assigned to group 0 by default
    - Group names can be specified explicitly or derived from the BAM filename

    Args:
        bam_file: Path to BAM file. Must be coordinate-sorted. Index file (.bai)
            is recommended for optimal performance but not required.
        output_dir: Directory path where the insertion count matrix will be saved.
            Will be created if it doesn't exist. Contains:
            - Matrix data files (val_data, idx_data, etc.)
            - chrom_offsets.json: Chromosome offset mapping
            - group_names.json: Group names for each pseudobulk
            - library_size.json: Library sizes per group
            - attrs.json: Metadata (store_type, source, etc.)
        cell_groups: Dictionary mapping cell barcode (CB tag value) to group ID.
            For bulk data without CB tags, use {"bulk.FILE_PREFIX": 0} where
            FILE_PREFIX is derived from the BAM filename.
            Example for single-cell::
                {"AAACCCAAGAAACCAT-1": 0, "AAACCCAAGAAACCAT-2": 1, ...}
            Example for bulk::
                {"bulk.sample1": 0}
        chrom_sizes: Chromosome sizes. Can be:
            - Path to UCSC-style chrom.sizes file (tab-separated: chr<tab>size)
            - Dictionary mapping chromosome names to sizes: {"chr1": 248956422, ...}
        shift_start: Basepairs to add to fragment start coordinates.
            Default is 4 for Tn5 transposase (ATAC-seq). Set to 0 to disable.
        shift_end: Basepairs to add to fragment end coordinates (can be negative).
            Default is -5 for Tn5 transposase (ATAC-seq). This accounts for:
            - Tn5 cut site offset (-4bp)
            - BED format convention (-1bp, since end is 1 past last base)
            Set to 0 to disable.
        threads: Number of parallel threads to use. Default is 0 (use all available CPUs).
            Each thread processes a separate chunk of the genome.
        group_names: Optional list of group names in the same order as group IDs.
            If not provided, group names are automatically derived from the BAM filename:
            - Single group: Uses filename prefix (e.g., "sample1" from "sample1.bam")
            - Multiple groups: Uses filename prefix with group ID suffix
              (e.g., "sample1_group_0", "sample1_group_1")
            Group names are used for :class:`CelltypeDenseBPCellsIO` compatibility.

    Returns:
        :class:`PrecalculatedInsertionMatrix`: A matrix object that can be used for
        downstream analysis or loaded with :class:`CelltypeDenseBPCellsIO`.

    Raises:
        ImportError: If pysam is not installed (required for cell discovery).
        RuntimeError: If BAM file cannot be read or processed.
        ValueError: If cell_groups is not a dictionary or chrom_sizes is invalid.

    Note:
        The current implementation is EXPERIMENTAL and will crash for matrices
        with more than 2^32-1 non-zero entries.

        The function performs a full scan of the BAM file to discover all cell
        barcodes before processing. This ensures correct group assignment but
        may be slow for very large files.

    Example:
        Single-cell ATAC-seq::
            >>> cell_groups = {
            ...     "AAACCCAAGAAACCAT-1": 0,  # Cell type A
            ...     "AAACCCAAGAAACCAT-2": 1,  # Cell type B
            ... }
            >>> chrom_sizes = {"chr1": 248956422, "chr2": 242193529}
            >>> matrix = precalculate_insertion_counts_bam(
            ...     bam_file="single_cell.bam",
            ...     output_dir="output",
            ...     cell_groups=cell_groups,
            ...     chrom_sizes=chrom_sizes
            ... )

        Bulk ATAC-seq::
            >>> matrix = precalculate_insertion_counts_bam(
            ...     bam_file="bulk_sample.bam",
            ...     output_dir="output",
            ...     cell_groups={"bulk": 0},
            ...     chrom_sizes=chrom_sizes,
            ...     group_names=["MySample"]  # Optional: custom group name
            ... )

    See Also:
        :func:`precalculate_insertion_counts`: Precalculate from BPCells fragments format
        :class:`PrecalculatedInsertionMatrix`: The returned matrix object
        :class:`CelltypeDenseBPCellsIO`: Load the output for analysis in Caesar
    """
    # Normalize BAM path
    bam_path = os.path.abspath(os.path.expanduser(bam_file))
    
    if not isinstance(cell_groups, dict):
        raise TypeError("cell_groups must be a dict mapping cell barcode (str) to group ID (int)")
    
    # Parse chrom_sizes
    if isinstance(chrom_sizes, str):
        chrom_sizes = pd.read_csv(chrom_sizes, sep="\t", names=["chrom", "size"])
        chrom_sizes = {t.chrom: t.size for t in chrom_sizes.itertuples()}
    
    # Extract file prefix for bulk barcode naming
    bam_basename = os.path.splitext(os.path.basename(bam_path))[0]
    # Remove common suffixes like .sorted, .dedup, etc.
    for suffix in ['.sorted', '.dedup', '.filtered', '.bam']:
        if bam_basename.endswith(suffix):
            bam_basename = bam_basename[:-len(suffix)]
    bulk_barcode = f"bulk.{bam_basename}"
    
    # OPTIMIZATION: Quick check if CB tags exist using C++ (faster than pysam)
    # If no CB tags, skip full scan and use bulk barcode directly
    import bpcells.cpp as cpp
    
    try:
        # Quick check using C++ (samples first 10k reads)
        has_cb_tags = cpp.bam_has_cb_tags(bam_path, "CB", 10000)
        
        # Get chromosome names quickly (just header read, very fast)
        import pysam
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            bam_chr_names = list(bam.references)
        
        if not has_cb_tags:
            # No CB tags - use bulk.FILE_PREFIX barcode for bulk data
            # Skip full scan - this saves significant time for bulk data
            cell_barcodes_ordered = [bulk_barcode]
            cell_barcode_set = {bulk_barcode}
        else:
            # Single-cell data - need to discover all cells
            # Use C++ for discovery (faster than pysam)
            cell_barcodes_ordered = cpp.discover_cells_from_bam(bam_path, "CB", "")
            cell_barcode_set = set(cell_barcodes_ordered)
    except ImportError:
        # Fallback to pysam if C++ functions not available
        import pysam
        try:
            with pysam.AlignmentFile(bam_path, "rb") as bam:
                bam_chr_names = list(bam.references)
                # Check first few reads to see if CB tags exist
                has_cb_tags = False
                sample_count = 0
                for read in bam:
                    if read.is_proper_pair and read.is_read1 and not read.is_unmapped:
                        if read.has_tag("CB"):
                            has_cb_tags = True
                            break
                        sample_count += 1
                        if sample_count >= 10000:  # Sample first 10k reads to check
                            break
                
                # Reset file pointer
                bam.close()
                bam = pysam.AlignmentFile(bam_path, "rb")
                
                if not has_cb_tags:
                    # No CB tags - use bulk.FILE_PREFIX barcode for bulk data
                    cell_barcodes_ordered = [bulk_barcode]
                    cell_barcode_set = {bulk_barcode}
                else:
                    # Read through entire BAM to discover all cell barcodes in order
                    cell_barcodes_ordered = []
                    cell_barcode_set = set()
                    for read in bam:
                        if read.is_proper_pair and read.is_read1 and not read.is_unmapped:
                            if read.has_tag("CB"):
                                cb_tag = read.get_tag("CB")
                                if cb_tag and cb_tag not in cell_barcode_set:
                                    cell_barcodes_ordered.append(cb_tag)
                                    cell_barcode_set.add(cb_tag)
        except ImportError:
            raise ImportError("pysam is required for precalculate_insertion_counts_bam. Install with: pip install pysam")
        except Exception as e:
            raise RuntimeError(f"Error reading BAM file: {e}")
    except Exception as e:
        raise RuntimeError(f"Error reading BAM file: {e}")
    
    # Create cell_groups array in discovered order
    # If no cells found in cell_groups dict, assign all to group 0
    cell_groups_array = np.full(len(cell_barcodes_ordered), -1, dtype=np.int32)
    for i, barcode in enumerate(cell_barcodes_ordered):
        if barcode in cell_groups:
            cell_groups_array[i] = cell_groups[barcode]
        else:
            # If barcode not in cell_groups, assign to group 0 (default)
            cell_groups_array[i] = 0
    
    # If all entries are -1 (no valid groups), assign all to group 0
    if np.all(cell_groups_array == -1):
        cell_groups_array.fill(0)
    
    # Re-order chrom_sizes to match BAM chromosome order
    chrom_sizes = dict(i for i in chrom_sizes.items() if i[0] in bam_chr_names)
    chrom_sizes = dict(sorted(chrom_sizes.items(), key=lambda x: bam_chr_names.index(x[0]) if x[0] in bam_chr_names else len(bam_chr_names)))
    
    # Determine group names before calling C++ (needed for row names in matrix)
    unique_group_ids = sorted(set(cell_groups.values()))
    
    # Determine group names:
    # IMPORTANT: For bulk data, cell_groups should contain {"bulk.FILEPREFIX": 0}
    # and group_names.json should use "bulk.FILEPREFIX" as the group name by default
    # 1. If group_names is a dict: map barcodes to group names (use barcode if not in dict)
    # 2. If group_names is a list: use as-is (one per unique group ID)
    # 3. Otherwise: use barcode names as group names (default - bulk.FILEPREFIX)
    if isinstance(group_names, dict):
        # Dict mapping barcodes to group names: {'bulk.FILE_PREFIX': 'GROUP_NAME'}
        # Map each unique group ID to its group name
        # IMPORTANT: If barcode not in dict, use barcode itself (bulk.FILEPREFIX)
        group_names_list = []
        for gid in unique_group_ids:
            # Find a barcode that maps to this group ID
            group_name = None
            barcode_found = None
            for barcode, bgid in cell_groups.items():
                if bgid == gid:
                    barcode_found = barcode
                    # For bulk barcodes (bulk.FILEPREFIX), always use the barcode itself
                    # This ensures group_names.json uses bulk.FILEPREFIX format
                    if barcode.startswith("bulk."):
                        group_name = barcode  # Always use bulk.FILEPREFIX for bulk data
                    else:
                        # For single-cell data, use mapping if provided, otherwise use barcode
                        group_name = group_names.get(barcode, barcode)
                    break
            if group_name is None:
                # Fallback: use barcode if found, otherwise use group ID as string
                group_name = barcode_found if barcode_found else str(gid)
            group_names_list.append(group_name)
    elif isinstance(group_names, list):
        # List of group names: validate length matches unique groups
        if len(group_names) != len(unique_group_ids):
            warnings.warn(
                f"group_names length ({len(group_names)}) does not match number of unique groups "
                f"({len(unique_group_ids)}). Using barcode names instead.",
                UserWarning,
                stacklevel=2
            )
            # Fall back to barcode-based naming
            group_names_list = []
            for gid in unique_group_ids:
                # Find a barcode that maps to this group ID
                group_name = None
                for barcode, bgid in cell_groups.items():
                    if bgid == gid:
                        group_name = barcode
                        break
                if group_name is None:
                    group_name = str(gid)
                group_names_list.append(group_name)
        else:
            # Map group_names to group IDs in sorted order
            group_id_to_idx = {gid: i for i, gid in enumerate(unique_group_ids)}
            group_names_list = [group_names[group_id_to_idx[gid]] for gid in unique_group_ids]
    else:
        # No group_names provided: use barcode names as group names
        group_names_list = []
        for gid in unique_group_ids:
            # Find a barcode that maps to this group ID (use first one found)
            group_name = None
            for barcode, bgid in cell_groups.items():
                if bgid == gid:
                    group_name = barcode
                    break
            if group_name is None:
                group_name = str(gid)
            group_names_list.append(group_name)
    
    # Ensure output directory doesn't exist (C++ will create it)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Use context manager to ensure temp directory stays alive during C++ execution
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Try calling with group_names first (new version), fall back to old signature if needed
        try:
            bpcells.cpp.precalculate_pseudobulk_coverage_bam(
                bam_path,
                output_dir,
                tmp_dir,
                list(chrom_sizes.keys()),
                list(chrom_sizes.values()),
                cell_groups_array.tolist(),
                shift_start,
                shift_end,
                1,  # bin_size hardcoded to 1 for insertion counts
                threads,
                group_names_list  # Pass group names to C++ for row names
            )
        except TypeError:
            # Old version without group_names parameter - C++ will use numeric names
            # Group names will still be saved in group_names.json by Python code below
            bpcells.cpp.precalculate_pseudobulk_coverage_bam(
                bam_path,
                output_dir,
                tmp_dir,
                list(chrom_sizes.keys()),
                list(chrom_sizes.values()),
                cell_groups_array.tolist(),
                shift_start,
                shift_end,
                1,  # bin_size hardcoded to 1 for insertion counts
                threads
            )
    
    # Save chrom_offsets
    chrom_offsets = dict(zip(chrom_sizes.keys(), [0] + np.cumsum(list(chrom_sizes.values()))[:-1].tolist()))
    json.dump(chrom_offsets, open(f"{output_dir}/chrom_offsets.json", "w"), indent=2)
    
    # Save group_names.json for CelltypeDenseBPCellsIO compatibility (backup/verification)
    group_names_path = os.path.join(output_dir, "group_names.json")
    with open(group_names_path, "w") as f:
        json.dump(group_names_list, f, indent=2)
    
    # Save attrs.json with metadata for Caesar compatibility
    # This helps CelltypeDenseBPCellsIO and other IO classes load metadata
    attrs = {
        "assembly": None,  # Could be inferred from chrom_sizes if needed
        "store_type": "bpcells_celltype_dense",  # Indicates this is celltype dense format
        "class": "CelltypeDenseBPCellsIO",  # For detect_store_type
        "source": "bam_file",  # Indicate this came from BAM
        "bam_file": bam_path,  # Store source BAM path (use normalized path)
        "shift_start": shift_start,
        "shift_end": shift_end,
        "group_names": group_names_list,  # Store group names for reference
    }
    attrs_path = os.path.join(output_dir, "attrs.json")
    with open(attrs_path, "w") as f:
        json.dump(attrs, f, indent=2)
    
    return PrecalculatedInsertionMatrix(output_dir)


def precalculate_insertion_counts_bam_multi(
    bam_files: List[str],
    output_dir: str,
    chrom_sizes: Union[str, Dict[str, int]],
    shift_start: int = 4,
    shift_end: int = -5,
    threads: int = 0,
    group_names: Optional[Union[List[str], Dict[str, str]]] = None
) -> PrecalculatedInsertionMatrix:
    """Precalculate per-base insertion counts from multiple BAM files in one call.

    This function processes multiple bulk BAM files (without CB tags) directly to BPCells
    celltype_dense format in a single matrix. Each BAM file gets a unique cell prefix
    (e.g., "bulk.FILENAME") to distinguish cells, allowing all BAMs to be processed together.

    Args:
        bam_files: List of paths to BAM files (must be coordinate-sorted)
        output_dir: Directory path where the insertion count matrix will be saved
        chrom_sizes: Chromosome sizes (dict or path to chrom.sizes file)
        shift_start: Basepairs to add to fragment start coordinates (default: 4)
        shift_end: Basepairs to add to fragment end coordinates (default: -5)
        threads: Number of parallel threads (0 = use all available CPUs)
        group_names: Optional list of group names, one per BAM file

    Returns:
        PrecalculatedInsertionMatrix: A matrix object with one row per BAM file

    Example:
        >>> matrix = precalculate_insertion_counts_bam_multi(
        ...     bam_files=["sample1.bam", "sample2.bam", "sample3.bam"],
        ...     output_dir="output",
        ...     chrom_sizes={"chr1": 248956422, "chr2": 242193529},
        ...     group_names=["Sample1", "Sample2", "Sample3"]  # Optional
        ... )
    """
    if not bam_files:
        raise ValueError("At least one BAM file required")
    
    # Normalize BAM paths
    bam_paths = [os.path.abspath(os.path.expanduser(bam_file)) for bam_file in bam_files]
    
    # Parse chrom_sizes
    if isinstance(chrom_sizes, str):
        chrom_sizes = pd.read_csv(chrom_sizes, sep="\t", names=["chrom", "size"])
        chrom_sizes = {t.chrom: t.size for t in chrom_sizes.itertuples()}
    
    # Extract file prefixes for cell prefixes
    bam_prefixes = []
    for bam_path in bam_paths:
        bam_basename = os.path.splitext(os.path.basename(bam_path))[0]
        # Remove common suffixes
        for suffix in ['.sorted', '.dedup', '.filtered', '.bam']:
            if bam_basename.endswith(suffix):
                bam_basename = bam_basename[:-len(suffix)]
        bam_prefixes.append(f"bulk.{bam_basename}")
    
    # Discover cells from all BAM files with prefixes
    import pysam
    
    all_cell_barcodes_ordered = []
    all_cell_barcode_set = set()
    bam_chr_names = None
    
    try:
        # Use first BAM to get chromosome names
        with pysam.AlignmentFile(bam_paths[0], "rb") as bam:
            bam_chr_names = list(bam.references)
        
        # Discover cells from each BAM file
        for bam_idx, bam_path in enumerate(bam_paths):
            prefix = bam_prefixes[bam_idx]
            
            with pysam.AlignmentFile(bam_path, "rb") as bam:
                # Check if CB tags exist
                has_cb_tags = False
                sample_count = 0
                for read in bam:
                    if read.is_proper_pair and read.is_read1 and not read.is_unmapped:
                        if read.has_tag("CB"):
                            has_cb_tags = True
                            break
                        sample_count += 1
                        if sample_count >= 10000:
                            break
                
                # Reset and discover cells
                bam.close()
                bam = pysam.AlignmentFile(bam_path, "rb")
                
                if not has_cb_tags:
                    # No CB tags - use prefixed "bulk.FILENAME" barcode
                    prefixed_barcode = prefix
                    if prefixed_barcode not in all_cell_barcode_set:
                        all_cell_barcodes_ordered.append(prefixed_barcode)
                        all_cell_barcode_set.add(prefixed_barcode)
                else:
                    # Has CB tags - prefix each barcode
                    for read in bam:
                        if read.is_proper_pair and read.is_read1 and not read.is_unmapped:
                            if read.has_tag("CB"):
                                cb_tag = read.get_tag("CB")
                                if cb_tag:
                                    prefixed_barcode = f"{prefix}.{cb_tag}"
                                    if prefixed_barcode not in all_cell_barcode_set:
                                        all_cell_barcodes_ordered.append(prefixed_barcode)
                                        all_cell_barcode_set.add(prefixed_barcode)
    except ImportError:
        raise ImportError("pysam is required for precalculate_insertion_counts_bam_multi. Install with: pip install pysam")
    except Exception as e:
        raise RuntimeError(f"Error reading BAM files: {e}")
    
    # Create cell_groups dict: each BAM file gets its own group (group ID = BAM index)
    cell_groups = {}
    for i, barcode in enumerate(all_cell_barcodes_ordered):
        # Determine which BAM this barcode belongs to based on prefix
        for bam_idx, prefix in enumerate(bam_prefixes):
            if barcode.startswith(prefix):
                cell_groups[barcode] = bam_idx
                break
        else:
            # Fallback: assign to group 0
            cell_groups[barcode] = 0
    
    # Create cell_groups array in discovered order
    cell_groups_array = np.array([cell_groups.get(barcode, 0) for barcode in all_cell_barcodes_ordered], dtype=np.int32)
    
    # Re-order chrom_sizes to match BAM chromosome order
    chrom_sizes = dict(i for i in chrom_sizes.items() if i[0] in bam_chr_names)
    chrom_sizes = dict(sorted(chrom_sizes.items(), key=lambda x: bam_chr_names.index(x[0]) if x[0] in bam_chr_names else len(bam_chr_names)))
    
    # Determine group names
    unique_group_ids = sorted(set(cell_groups_array.tolist()))
    
    # Determine group names:
    # 1. If group_names is a dict: map barcodes (bulk.FILE_PREFIX) to group names
    # 2. If group_names is a list: use as-is (one per BAM file/group ID)
    # 3. Otherwise: use barcode prefixes as group names (default)
    if isinstance(group_names, dict):
        # Dict mapping barcodes to group names: {'bulk.FILE_PREFIX': 'GROUP_NAME'}
        group_names_list = []
        for gid in unique_group_ids:
            # Find the barcode prefix for this group ID (should be bam_prefixes[gid])
            if gid < len(bam_prefixes):
                barcode_prefix = bam_prefixes[gid]
                # For bulk barcodes (bulk.FILEPREFIX), always use the barcode itself
                # This ensures group_names.json uses bulk.FILEPREFIX format
                if barcode_prefix.startswith("bulk."):
                    group_name = barcode_prefix  # Always use bulk.FILEPREFIX for bulk data
                else:
                    # For single-cell data, use mapping if provided, otherwise use barcode prefix
                    group_name = group_names.get(barcode_prefix, barcode_prefix)
            else:
                group_name = str(gid)
            group_names_list.append(group_name)
    elif isinstance(group_names, list):
        # List of group names: validate length matches number of BAM files
        if len(group_names) != len(bam_files):
            warnings.warn(
                f"group_names length ({len(group_names)}) does not match number of BAM files "
                f"({len(bam_files)}). Using barcode prefixes instead.",
                UserWarning,
                stacklevel=2
            )
            group_names_list = [bam_prefixes[i] for i in unique_group_ids]
        else:
            # Map group_names to group IDs
            group_id_to_idx = {gid: i for i, gid in enumerate(unique_group_ids)}
            group_names_list = [group_names[group_id_to_idx[gid]] for gid in unique_group_ids]
    else:
        # No group_names provided: use barcode prefixes as group names
        group_names_list = [bam_prefixes[i] for i in unique_group_ids]
    
    # Ensure output directory doesn't exist
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Check if C++ function supports multi-BAM (will be added)
    # For now, we'll need to use the single-BAM approach with MergeFragments
    # This requires implementing the multi-BAM C++ function first
    
    # Use context manager to ensure temp directory stays alive during C++ execution
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Call C++ multi-BAM function
        bpcells.cpp.precalculate_pseudobulk_coverage_bam_multi(
            bam_paths,
            bam_prefixes,
            output_dir,
            tmp_dir,
            list(chrom_sizes.keys()),
            list(chrom_sizes.values()),
            cell_groups_array.tolist(),
            shift_start,
            shift_end,
            1,  # bin_size hardcoded to 1 for insertion counts
            threads,
            group_names_list  # Pass group names to C++ for row names
        )
    
    # Save chrom_offsets
    chrom_offsets = dict(zip(chrom_sizes.keys(), [0] + np.cumsum(list(chrom_sizes.values()))[:-1].tolist()))
    json.dump(chrom_offsets, open(f"{output_dir}/chrom_offsets.json", "w"), indent=2)
    
    # Save group_names.json for CelltypeDenseBPCellsIO compatibility
    group_names_path = os.path.join(output_dir, "group_names.json")
    with open(group_names_path, "w") as f:
        json.dump(group_names_list, f, indent=2)
    
    # Save attrs.json with metadata for Caesar compatibility
    attrs = {
        "assembly": None,
        "store_type": "bpcells_celltype_dense",
        "class": "CelltypeDenseBPCellsIO",
        "source": "bam_files_multi",
        "bam_files": bam_paths,
        "shift_start": shift_start,
        "shift_end": shift_end,
        "group_names": group_names_list,
    }
    attrs_path = os.path.join(output_dir, "attrs.json")
    with open(attrs_path, "w") as f:
        json.dump(attrs, f, indent=2)
    
    return PrecalculatedInsertionMatrix(output_dir)

