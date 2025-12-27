# Copyright 2023 BPCells contributors
# 
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

import bpcells
import bpcells.cpp

import json
import logging
import tempfile
import os.path
import warnings

from typing import Dict, List, Optional, Tuple, Union
import sys
if sys.version_info >= (3, 9):
    from collections.abc import Sequence
else:
    from typing import Sequence

import numpy as np
import pandas as pd

# Import DirMatrix for binned matrix storage
try:
    from .matrix import DirMatrix
except ImportError:
    # Fallback if matrix module is not available
    DirMatrix = None

# Set up logger for build_cell_groups
_logger = logging.getLogger(__name__)

def _extract_valid_group_names(cell_groups: pd.Categorical, group_order: Optional[Sequence[str]] = None) -> List[str]:
    """Extract valid group names from a categorical, preserving order from group_order.
    
    This function ensures that only groups that actually have cells are included,
    and preserves the order from group_order if provided. This is critical for
    maintaining consistency between group_names, library_size, and matrix shape.
    
    Args:
        cell_groups: pd.Categorical with group assignments
        group_order: Optional original group_order used to create the categorical.
            If provided, preserves this order for groups that have cells.
    
    Returns:
        List of group names that actually have cells, in order from group_order
        (or order of appearance if group_order not provided).
    """
    # Get unique groups that actually have cells (non-NaN values)
    if hasattr(cell_groups, 'filtered_cell_indices'):
        # For filtered categoricals, get the actual values
        valid_groups = list(dict.fromkeys(cell_groups))
    else:
        # For full categoricals, get unique non-NaN values
        valid_groups = [g for g in cell_groups.categories if g in cell_groups.dropna().unique()]
    
    # If group_order is provided, preserve its order for groups that have cells
    if group_order is not None:
        # Create ordered list: groups in group_order that are in valid_groups
        ordered_valid = [g for g in group_order if g in valid_groups]
        # Add any remaining valid groups not in group_order (shouldn't happen, but be safe)
        remaining = [g for g in valid_groups if g not in ordered_valid]
        return ordered_valid + remaining
    
    # Otherwise, return in order of appearance
    return valid_groups

def import_10x_fragments(input: str, output: str, shift_start: int = 0, shift_end: int = 0, keeper_cells: Optional[List[str]] = None):
    """Convert 10x fragment file to BPCells format

    Args:
        input (str): Path to 10x input file
        output (str): Path to BPCells output directory
        shift_start (int): Basepairs to add to start coordinates (generally positive number)
        shift_end (int): Basepairs to subtract from end coordinates (generally negative number)
        keeper_cells (list[str]): If not None, only save fragments from cells in the keeper_cells list
    """
    keeper_cells = np.asarray(keeper_cells) if keeper_cells is not None else keeper_cells
    bpcells.cpp.import_10x_fragments(input, output, shift_start, shift_end, keeper_cells)

def build_cell_groups(
    fragments: Union[str, List[str]], 
    cell_ids: Union[Sequence[str], Dict[str, Sequence[str]]], 
    group_ids: Union[Sequence[str], Dict[str, Sequence[str]]], 
    group_order: Sequence[str],
    min_library_size: int = 1000,
    max_library_size: int = 50000
) -> pd.Categorical:
    """Build cell_groups categorical for use in :func:`pseudobulk_insertion_counts()`

    Args:
        fragments (str | list[str]): Path to BPCells fragments directory, or list of paths to multiple fragment directories
        cell_ids (list[str] | dict[str, list[str]]): 
            - If list: List of cell IDs. **Only supported for single fragment files.**
              Will raise a warning if used with multiple fragments.
            - If dict: Mapping from fragment path to list of cell IDs for that fragment.
              Keys must match paths in ``fragments``. **Required for multiple fragments**
              to clearly specify which cells belong to which fragment file.
        group_ids (list[str] | dict[str, list[str]]): 
            - If list: List of pseudobulk IDs for each cell (same length as ``cell_ids``).
              **Only supported for single fragment files.**
            - If dict: Mapping from fragment path to list of group IDs for that fragment.
              Keys must match paths in ``fragments``. Each value must have same length as
              corresponding ``cell_ids[path]``. **Required for multiple fragments.**
        group_order (list[str]): Output order of pseudobulks (must contain all unique ``group_ids``)
        min_library_size (int): Minimum library size (fragment count) to include a cell. 
            Cells with library size < min_library_size will be excluded (set to NaN).
            Default: 1000
        max_library_size (int): Maximum library size (fragment count) to include a cell.
            Cells with library size > max_library_size will be excluded (set to NaN).
            Default: 50000

    Returns:
        pd.Categorical:
        Pandas Categorical suitable as input for ``cell_groups`` in :func:`precalculate_insertion_counts()`.
        The categorical has the following properties:
        - Length equals total number of cells in the ``fragments`` input
        - Contains group assignments for each cell (or NaN if the cell is excluded from consideration)
        - Categories include ONLY groups that have at least one cell after filtering
        - Categories are ordered according to ``group_order`` (groups not in group_order are appended)
        - When using multiple fragment files, the categorical covers all cells across all files
        
        The categorical has two attributes:
        - ``filtered_cell_indices``: Array of global cell indices that passed filtering
        - ``original_group_order``: Original group_order used to create the categorical (for consistency)
        
        **Important**: The categorical only includes categories for groups that actually have cells.
        This ensures consistency between group_names, library_size, and matrix shape in the output.

    Examples:
        Single fragment (list-based API):
        >>> cell_groups = build_cell_groups(
        ...     fragments="/path/to/fragments",
        ...     cell_ids=["cell1", "cell2", "cell3"],
        ...     group_ids=["groupA", "groupA", "groupB"],
        ...     group_order=["groupA", "groupB"]
        ... )

        Multiple fragments (dict-based API - recommended):
        >>> cell_groups = build_cell_groups(
        ...     fragments=["/path/to/frag1", "/path/to/frag2"],
        ...     cell_ids={
        ...         "/path/to/frag1": ["cell1", "cell2"],
        ...         "/path/to/frag2": ["cell3", "cell4"]
        ...     },
        ...     group_ids={
        ...         "/path/to/frag1": ["groupA", "groupA"],
        ...         "/path/to/frag2": ["groupB", "groupB"]
        ...     },
        ...     group_order=["groupA", "groupB"]
        ... )

    See Also:
        :func:`pseudobulk_insertion_counts`
    """
    _logger.info("=" * 80)
    _logger.info("build_cell_groups: Starting cell group construction")
    _logger.info(f"  Fragments: {len(fragments) if isinstance(fragments, list) else 1} fragment(s)")
    _logger.info(f"  Library size filter: {min_library_size} <= lib_size <= {max_library_size}")
    _logger.info(f"  Group order: {len(group_order)} groups")
    
    # Convert single path to list for uniform handling
    if isinstance(fragments, str):
        fragments = [fragments]
    
    # Normalize fragment paths to absolute paths for consistent matching
    fragments_normalized = [os.path.abspath(os.path.expanduser(f)) for f in fragments]
    _logger.debug(f"  Normalized fragment paths: {[os.path.basename(f) for f in fragments_normalized]}")

    # Determine if using dict-based API (for multiple fragments) or list-based API
    using_dict_api = isinstance(cell_ids, dict)
    _logger.info(f"  API mode: {'dict-based' if using_dict_api else 'list-based'}")
    
    # Warn if list-based API is used with multiple fragments
    if not using_dict_api and len(fragments_normalized) > 1:
        _logger.warning(
            "List-based API (cell_ids as list) used with multiple fragments. "
            "This may cause incorrect cell-to-fragment mapping. Consider using dict-based API."
        )
        warnings.warn(
            "List-based API (cell_ids as list) should only be used with a single fragment file. "
            "For multiple fragments, use the dict-based API where cell_ids and group_ids are dicts "
            "mapping fragment paths to cell/group lists. This ensures correct cell-to-fragment "
            "mapping and avoids ordering errors.",
            UserWarning,
            stacklevel=2
        )
    
    if using_dict_api:
        # Dict-based API: cell_ids and group_ids are dicts mapping fragment path -> list
        if not isinstance(group_ids, dict):
            raise TypeError("When cell_ids is a dict, group_ids must also be a dict")
        
        # Normalize dict keys to absolute paths
        cell_ids_normalized = {os.path.abspath(os.path.expanduser(k)): v for k, v in cell_ids.items()}
        group_ids_normalized = {os.path.abspath(os.path.expanduser(k)): v for k, v in group_ids.items()}
        
        # Validate that all fragment paths have corresponding entries
        missing_frags = set(fragments_normalized) - set(cell_ids_normalized.keys())
        if missing_frags:
            _logger.error(f"Missing cell_ids entries for fragments: {missing_frags}")
            raise ValueError(f"Missing cell_ids entries for fragments: {missing_frags}")
        missing_frags = set(fragments_normalized) - set(group_ids_normalized.keys())
        if missing_frags:
            _logger.error(f"Missing group_ids entries for fragments: {missing_frags}")
            raise ValueError(f"Missing group_ids entries for fragments: {missing_frags}")
        
        # Validate lengths match for each fragment
        for frag_path in fragments_normalized:
            cell_count = len(cell_ids_normalized[frag_path])
            group_count = len(group_ids_normalized[frag_path])
            if cell_count != group_count:
                _logger.error(
                    f"Fragment {os.path.basename(frag_path)}: cell_ids length ({cell_count}) != "
                    f"group_ids length ({group_count})"
                )
                raise ValueError(
                    f"cell_ids and group_ids must have same length for fragment {frag_path}. "
                    f"Got {cell_count} and {group_count}"
                )
            _logger.debug(f"  Fragment {os.path.basename(frag_path)}: {cell_count} cells")
    else:
        # List-based API: cell_ids and group_ids are sequences
        if isinstance(group_ids, dict):
            _logger.error("Type mismatch: cell_ids is list but group_ids is dict")
            raise TypeError("When cell_ids is a list, group_ids must also be a list")
        if len(cell_ids) != len(group_ids):
            _logger.error(f"Length mismatch: cell_ids ({len(cell_ids)}) != group_ids ({len(group_ids)})")
            raise ValueError(f"cell_ids and group_ids must have same length. Got {len(cell_ids)} and {len(group_ids)}")
        _logger.debug(f"  List-based API: {len(cell_ids)} cells")

    # Build cell index lookup across all fragment files
    # Track cells in order across fragments to preserve fragment context for duplicate barcodes
    # Sequential matching is O(total_cells) which is optimal since we iterate through all cells anyway
    _logger.info("Loading cells from fragment files...")
    cell_sequence = []  # List of (frag_path, cell_name, global_index, frag_local_index) in order across all fragments
    current_index = 0
    frag_cell_counts = {}
    for frag_path in fragments_normalized:
        frag_local_index = 0
        frag_cells = list(bpcells.cpp.cell_names_fragments_dir(frag_path))
        frag_cell_counts[frag_path] = len(frag_cells)
        for cell_name in frag_cells:
            cell_sequence.append((frag_path, cell_name, current_index, frag_local_index))
            current_index += 1
            frag_local_index += 1
        _logger.debug(f"  Fragment {os.path.basename(frag_path)}: {len(frag_cells)} cells")

    # Total number of cells across all fragments
    total_cells = current_index
    _logger.info(f"Total cells across all fragments: {total_cells}")

    # Load library sizes from each fragment directory
    _logger.info("Loading library sizes for filtering...")
    frag_library_sizes = {}  # Map from frag_path to array of library sizes
    for frag_path in fragments_normalized:
        library_size_path = os.path.join(frag_path, "library_size.json")
        if os.path.exists(library_size_path):
            try:
                with open(library_size_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "library_sizes" in data:
                        frag_library_sizes[frag_path] = np.array(data["library_sizes"], dtype=np.uint64)
                        _logger.debug(
                            f"  Fragment {os.path.basename(frag_path)}: loaded {len(frag_library_sizes[frag_path])} library sizes"
                        )
                    else:
                        _logger.info(f"  Fragment {os.path.basename(frag_path)}: invalid library_size.json format")
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                _logger.info(
                    f"  Fragment {os.path.basename(frag_path)}: could not load library sizes: {e}. "
                    f"Library size filtering will be skipped for this fragment."
                )
                warnings.warn(
                    f"Could not load library sizes from {library_size_path}: {e}. "
                    f"Library size filtering will be skipped for this fragment.",
                    UserWarning,
                    stacklevel=2
                )
        else:
            _logger.debug(f"  Fragment {os.path.basename(frag_path)}: no library_size.json found, skipping library size filter")
    
    if frag_library_sizes:
        _logger.info(f"Library sizes loaded for {len(frag_library_sizes)}/{len(fragments_normalized)} fragments")
    else:
        _logger.info("No library sizes found - library size filtering will be skipped")

    # Create array of group assignments - only include cells that pass filters
    # Track which cells are included (filtered_cell_indices) and their group assignments
    _logger.info("Matching cells and applying filters...")
    filtered_cell_groups = []
    filtered_cell_indices = []  # Maps filtered index -> global cell index in fragments
    
    # Statistics for logging
    stats_matched = 0
    stats_unmatched = 0
    stats_filtered_libsize = 0
    stats_by_group = {}

    if using_dict_api:
        # Dict-based API: match cells per fragment
        # Create lookup maps for each fragment
        frag_cell_maps = {}
        for frag_path in fragments_normalized:
            # Create a set for fast lookup, but preserve order for validation
            cell_list = list(cell_ids_normalized[frag_path])
            group_list = list(group_ids_normalized[frag_path])
            frag_cell_maps[frag_path] = dict(zip(cell_list, group_list))
            
            # Validate that all group_ids are in group_order
            unique_groups = set(group_list)
            if not unique_groups <= set(group_order):
                missing = unique_groups - set(group_order)
                _logger.error(f"Fragment {os.path.basename(frag_path)}: groups not in group_order: {missing}")
                raise ValueError(f"group_ids contains groups not in group_order: {missing}")
            _logger.debug(f"  Fragment {os.path.basename(frag_path)}: {len(cell_list)} cells, {len(unique_groups)} unique groups")
        
        # Match cells using fragment-specific lookup
        for frag_path, cell_name, global_idx, frag_local_idx in cell_sequence:
            if frag_path in frag_cell_maps and cell_name in frag_cell_maps[frag_path]:
                # Match found - check library size filter if available
                group_id = frag_cell_maps[frag_path][cell_name]
                stats_matched += 1
                
                # Apply library size filter if library sizes are available
                if frag_path in frag_library_sizes:
                    if frag_local_idx < len(frag_library_sizes[frag_path]):
                        lib_size = frag_library_sizes[frag_path][frag_local_idx]
                        if lib_size < min_library_size or lib_size > max_library_size:
                            # Library size outside range - skip this cell entirely
                            stats_filtered_libsize += 1
                            continue
                    else:
                        _logger.info(
                            f"  Cell {cell_name} at index {frag_local_idx} >= library_size array length "
                            f"({len(frag_library_sizes[frag_path])}) for fragment {os.path.basename(frag_path)}"
                        )
                
                # Cell passes filter - include it
                filtered_cell_groups.append(group_id)
                filtered_cell_indices.append(global_idx)
                stats_by_group[group_id] = stats_by_group.get(group_id, 0) + 1
            else:
                stats_unmatched += 1
    else:
        # List-based API: name-based lookup (backward compatible)
        # Validate that all group_ids are in group_order
        unique_groups = set(group_ids)
        if not unique_groups <= set(group_order):
            missing = unique_groups - set(group_order)
            _logger.error(f"Groups not in group_order: {missing}")
            raise ValueError(f"group_ids contains groups not in group_order: {missing}")
        _logger.debug(f"  List-based API: {len(cell_ids)} cells, {len(unique_groups)} unique groups")
        
        # Create name-based lookup: cell_id -> group_id
        # This preserves backward compatibility - cells can be in any order
        # For single fragment, this works correctly
        # For multiple fragments with list-based API, we warn but still use name-based lookup
        # (duplicate barcodes across fragments will get the same group_id from the first match)
        cell_to_group = dict(zip(cell_ids, group_ids))
        
        # Match cells using name-based lookup
        for frag_path, cell_name, global_idx, frag_local_idx in cell_sequence:
            if cell_name in cell_to_group:
                # Match found - check library size filter if available
                group_id = cell_to_group[cell_name]
                stats_matched += 1
                
                # Apply library size filter if library sizes are available
                if frag_path in frag_library_sizes:
                    if frag_local_idx < len(frag_library_sizes[frag_path]):
                        lib_size = frag_library_sizes[frag_path][frag_local_idx]
                        if lib_size < min_library_size or lib_size > max_library_size:
                            # Library size outside range - skip this cell entirely
                            stats_filtered_libsize += 1
                            continue
                    else:
                        _logger.info(
                            f"  Cell {cell_name} at index {frag_local_idx} >= library_size array length "
                            f"({len(frag_library_sizes[frag_path])}) for fragment {os.path.basename(frag_path)}"
                        )
                
                # Cell passes filter - include it
                filtered_cell_groups.append(group_id)
                filtered_cell_indices.append(global_idx)
                stats_by_group[group_id] = stats_by_group.get(group_id, 0) + 1
            else:
                stats_unmatched += 1

    # Log matching statistics
    _logger.info("Cell matching statistics:")
    _logger.info(f"  Matched cells: {stats_matched}/{total_cells} ({100*stats_matched/total_cells:.1f}%)")
    if stats_unmatched > 0:
        _logger.info(f"  Unmatched cells: {stats_unmatched} (cells in fragments but not in cell_ids)")
    if stats_filtered_libsize > 0:
        _logger.info(f"  Filtered by library size: {stats_filtered_libsize} cells")
    _logger.info(f"  Final filtered cells: {len(filtered_cell_groups)}")
    
    if len(filtered_cell_groups) == 0:
        _logger.error("No cells passed filtering! Check cell_ids, group_ids, and library size filters.")
        raise ValueError(
            "No cells passed filtering. This can happen if:\n"
            "  1. No cells in cell_ids match cells in fragments\n"
            "  2. All cells were filtered out by library size constraints\n"
            f"  Library size filter: {min_library_size} <= lib_size <= {max_library_size}"
        )
    
    # Determine which groups actually have cells after filtering
    unique_filtered_groups = list(dict.fromkeys(filtered_cell_groups))
    
    # Only include categories that actually have cells, but preserve order from group_order
    # This ensures consistency: categories match the groups that will be in the output matrix
    valid_categories = [g for g in group_order if g in unique_filtered_groups]
    # Add any groups not in group_order (shouldn't happen if validation worked, but be safe)
    remaining = [g for g in unique_filtered_groups if g not in valid_categories]
    final_categories = valid_categories + remaining
    
    if remaining:
        _logger.info(f"  {len(remaining)} groups not in group_order but have cells")
    
    # Log group statistics
    _logger.info("Group statistics after filtering:")
    total_cells_in_groups = sum(stats_by_group.get(group, 0) for group in final_categories)
    _logger.info(f"  {len(final_categories)} groups with {total_cells_in_groups} total cells")
    
    # Check for groups in group_order that have no cells
    empty_groups = [g for g in group_order if g not in final_categories]
    if empty_groups:
        _logger.info(
            f"  {len(empty_groups)} groups in group_order with no cells (will be excluded from output)"
        )
    
    # Create categorical with only valid categories (groups that have cells)
    # Store filtered_cell_indices and original group_order as attributes for use in precalculate_insertion_counts
    _logger.info(f"Creating categorical with {len(final_categories)} groups (from {len(group_order)} in group_order)")
    cat = pd.Categorical(filtered_cell_groups, categories=final_categories, ordered=True)
    cat.filtered_cell_indices = np.array(filtered_cell_indices, dtype=np.int32)
    cat.original_group_order = list(group_order)  # Store for later use in precalculate_insertion_counts
    
    _logger.info("=" * 80)
    _logger.info(f"build_cell_groups: Complete - {len(filtered_cell_groups)} cells in {len(final_categories)} groups")
    
    return cat

def pseudobulk_insertion_counts(fragments: str, regions: pd.DataFrame, cell_groups: Union[Sequence[int], pd.Categorical], bin_size: int = 1) -> np.ndarray:
    """Calculate a pseudobulk coverage matrix

    Coverage is calculated as the number of start/end coordinates falling into a given position bin.

    Args:
        fragments (str): Path to BPCells fragments directory
        regions (pandas.DataFrame): Pandas dataframe with columns (``chrom``, ``start``, ``end``) representing
          genomic ranges (0-based, end-exclusive like BED format). All regions must be the same size.
          ``chrom`` should be a string column; ``start``/``end`` should be numeric.
        cell_groups (list[int] or pd.Categorical): List of pseudbulk groupings as created by :func:`build_cell_groups()`.
          If pd.Categorical, group names are taken from the categories.
        bin_size (int): Size for bins within each region given in basepairs. If the region width is not
          an even multiple of ``resolution_bp``, then the last region may be truncated.
    
    Returns:
        numpy.ndarray: Numpy array with dimensions (region, psudobulks, position) and type numpy.int32
    
    See Also: 
        :func:`build_cell_groups`
    """
    # Convert pd.Categorical to integer array if needed
    if isinstance(cell_groups, pd.Categorical):
        # Convert to integer codes, with -1 for NaN values
        cell_groups_array = cell_groups.codes.copy()
        cell_groups_array[cell_groups_array == -1] = -1  # Ensure NaN becomes -1
    else:
        cell_groups_array = np.asarray(cell_groups)
    
    chrs = bpcells.cpp.chr_names_fragments_dir(fragments)

    peak_order = sorted(
        range(regions.shape[0]),
        key = lambda i: (chrs.index(regions["chrom"].iloc[i]), regions["start"].iloc[i])
    )

    regions = regions.iloc[peak_order,]

    mat = bpcells.cpp.pseudobulk_coverage(
        fragments,
        np.asarray(regions["chrom"]),
        np.asarray(regions["start"]),
        np.asarray(regions["end"]),
        cell_groups_array,
        bin_size
    )
    return mat.reshape((mat.shape[0], -1, regions.shape[0]), order="F").transpose(2,0,1)


class PrecalculatedInsertionMatrix:
    """
    Disk-backed precalculated insertion matrix

    This reads per-base precalculated insertion matrices. The current implementation is EXPERIMENTAL, and will crash for matrices with more than
    2^32-1 non-zero entries.

    Args:
        path (str or list[str]): Path of the matrix directory, or list of matrix directories to combine
        
    See Also:
        :func:`precalculate_insertion_counts`
    """
    def __init__(self, path: Union[str, Sequence[str]]):
        if isinstance(path, str):
            # Single matrix mode
            self._paths = [str(os.path.abspath(os.path.expanduser(path)))]
            self._single_mode = True
        else:
            # Multiple matrix mode
            self._paths = [str(os.path.abspath(os.path.expanduser(p))) for p in path]
            self._single_mode = False
        
        # Load chrom_offsets from the first matrix (should be the same for all)
        self._chrom_offsets = json.load(open(f"{self._paths[0]}/chrom_offsets.json"))
        
        # Load library sizes from all matrices
        self._library_sizes = []
        self._all_group_names = []
        for p in self._paths:
            lib_size = self._load_library_sizes_single(p)
            if lib_size is not None:
                self._library_sizes.append(lib_size)
            else:
                self._library_sizes.append(np.array([]))
            
            # Get group names for this matrix
            group_names = self._get_group_names_single(p)
            self._all_group_names.extend(group_names)
        
        # Combine library sizes
        if all(len(ls) > 0 for ls in self._library_sizes):
            self._library_size = np.concatenate(self._library_sizes)
        else:
            self._library_size = np.array([])
        
        # Calculate combined shape
        if self._single_mode:
            first_shape = self._get_shape_single(self._paths[0])
            self._combined_shape = first_shape
        else:
            total_pseudobulks = sum(self._get_shape_single(p)[0] for p in self._paths)
            genome_size = self._get_shape_single(self._paths[0])[1]  # Should be same for all
            self._combined_shape = (total_pseudobulks, genome_size)

    def _get_shape_single(self, path: str) -> Tuple[int, int]:
        """Get shape for a single matrix"""
        return tuple(np.fromfile(f"{path}/shape", np.uint32, 2, offset=8))
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self._combined_shape

    @property
    def library_size(self) -> np.ndarray:
        return self._library_size
    
    def _get_group_names_single(self, path: str) -> List[str]:
        """Get group names for a single matrix"""
        # Try to get row names from the matrix
        try:
            # Read row names from the stored matrix
            row_names = bpcells.cpp.row_names_stored_matrix(path)
            if row_names and len(row_names) > 0 and any(name for name in row_names):
                return row_names
        except:
            pass
        
        # Fall back to JSON file if matrix doesn't have row names
        group_names_path = os.path.join(path, "group_names.json")
        if os.path.exists(group_names_path):
            with open(group_names_path, 'r') as f:
                return json.load(f)
        else:
            # Fall back to numeric names if neither exists
            shape = self._get_shape_single(path)
            return [str(i) for i in range(shape[0])]
    
    @property
    def group_names(self) -> List[str]:
        """Return the group names for each pseudobulk

        Returns:
            list: List of group names in the same order as library_size
        """
        return self._all_group_names

    def __repr__(self):
        if self._single_mode:
            return f"<PrecalculatedInsertionMatrix with {self.shape[0]} pseudobulks and {len(self._chrom_offsets)} chromosomes stored in \n\t{self._paths[0]}"
        else:
            return f"<PrecalculatedInsertionMatrix with {self.shape[0]} total pseudobulks from {len(self._paths)} matrices>"

    def get_counts(self, regions: pd.DataFrame):
        """Load pseudobulk insertion counts

        Args:
            regions (pandas.DataFrame): Pandas dataframe with columns (``chrom``, ``start``, ``end``) representing
                genomic ranges (0-based, end-exclusive like BED format). All regions must be the same size.
                ``chrom`` should be a string column; ``start``/``end`` should be numeric.
        
        Returns:
            numpy.ndarray: Numpy array of dimensions (region, psudobulks, position) and type numpy.int32
        """
        region_size = regions.end.iloc[0] - regions.start.iloc[0]
        assert (regions.end - regions.start == region_size).all()
        
        start_indices = [
            self._chrom_offsets[t.chrom] + t.start for t in regions.itertuples()
        ]
        
        if self._single_mode:
            # Single matrix mode
            return bpcells.cpp.query_precalculated_pseudobulk_coverage(
                self._paths[0],
                start_indices,
                region_size
            )\
                .reshape((region_size, regions.shape[0], -1), order="F")\
                .transpose((1,2,0))
        else:
            # Multiple matrix mode - get counts from each matrix and concatenate
            counts_list = []
            for path in self._paths:
                counts = bpcells.cpp.query_precalculated_pseudobulk_coverage(
                    path,
                    start_indices,
                    region_size
                )\
                    .reshape((region_size, regions.shape[0], -1), order="F")\
                    .transpose((1,2,0))
                counts_list.append(counts)
            
            # Concatenate along pseudobulks dimension (axis 1)
            return np.concatenate(counts_list, axis=1)

    def _load_library_sizes_single(self, path: str) -> np.ndarray:
        """Load library sizes from a single matrix directory"""
        # Try JSON format first (new format)
        json_filepath = os.path.join(path, "library_size.json")
        if os.path.exists(json_filepath):
            with open(json_filepath, 'r') as f:
                data = json.load(f)
                
                # Handle enhanced JSON format (v1.1+) with rowSums optimization
                if isinstance(data, dict):
                    if "library_sizes" in data:
                        return np.array(data["library_sizes"], dtype=np.uint64)
                    else:
                        raise ValueError("Invalid enhanced JSON format: missing library_sizes")
                
                else:
                    raise ValueError("Invalid JSON format: expected dict or list")
        
        # Fall back to binary format (legacy)
        binary_filepath = os.path.join(path, "library_size")
        if os.path.exists(binary_filepath):
            with open(binary_filepath, 'rb') as f:
                # Read the number of groups (uint32)
                size_bytes = f.read(4)
                if len(size_bytes) < 4:
                    raise ValueError("Invalid library size file format")
                size = np.frombuffer(size_bytes, dtype=np.uint32)[0]
                
                # Read the library sizes (uint64 for each group)
                data_bytes = f.read(size * 8)  # 8 bytes per uint64
                if len(data_bytes) < size * 8:
                    raise ValueError("Invalid library size file format")
                return np.frombuffer(data_bytes, dtype=np.uint64)
        
        return None
    
    
    def load_library_sizes(self) -> np.ndarray:
        """Load library sizes (total insertion counts) - backward compatibility method

        Returns:
            numpy.ndarray: Array of library sizes (one value per group)
        """
        return self._library_size

def precalculate_insertion_counts(fragments: Union[str, List[str]], output_dir: str, cell_groups: Union[Sequence[int], pd.Categorical],
                                 chrom_sizes: Union[str, Dict[str, int]], threads: int = 0,
                                 group_names: Optional[List[str]] = None):
    """Precalculate per-base insertion counts from fragment data

    The current implementation is EXPERIMENTAL, and will crash for matrices with more than
    2^32-1 non-zero entries.

    Args:
        fragments (str | list[str]): Path to a BPCells fragments directory, or list of paths to multiple fragment directories
        output_dir (str): Path to save the insertion counts in
        cell_groups (list[int] or pd.Categorical): Pseudobulk groupings as created by :func:`build_cell_groups()`.
            Should be the output of :func:`build_cell_groups()` to ensure correct cell-to-fragment mapping.
            When using multiple fragment files, the cell_groups should index across all files combined
            (e.g., if file1 has 100 cells and file2 has 150 cells, cell_groups should have length 250).
            If pd.Categorical, group names are taken from the categories.
            
            **Important**: When using multiple fragments, create `cell_groups` using the dict-based API
            in :func:`build_cell_groups()` to ensure correct mapping:
            
            >>> cell_groups = build_cell_groups(
            ...     fragments=["/path/to/frag1", "/path/to/frag2"],
            ...     cell_ids={"/path/to/frag1": [...], "/path/to/frag2": [...]},
            ...     group_ids={"/path/to/frag1": [...], "/path/to/frag2": [...]},
            ...     group_order=["groupA", "groupB"]
            ... )
            >>> precalculate_insertion_counts(
            ...     fragments=["/path/to/frag1", "/path/to/frag2"],
            ...     cell_groups=cell_groups,
            ...     ...
            ... )
        chrom_sizes (str | dict[str, int]): Path/URL of UCSC-style chrom.sizes file, or dictionary mapping chromosome names to sizes
        threads (int): Number of threads to use during matrix calculation (default = 1)
        group_names (list[str], optional): Names for each group in the same order as group indices (0, 1, 2, ...).
            Ignored if cell_groups is pd.Categorical.

    Returns:
        A :class:`PrecalculatedInsertionMatrix` object

    See Also:
        :func:`build_cell_groups` : Create cell_groups categorical from fragments
        :class:`PrecalculatedInsertionMatrix`
    """
    # Convert single path to list for uniform handling
    if isinstance(fragments, str):
        fragments = [fragments]
    
    # Normalize fragment paths to absolute paths for consistency
    fragments_normalized = [os.path.abspath(os.path.expanduser(f)) for f in fragments]
    
    # Validate that cell_groups length matches total cells across fragments
    # OR that it's a filtered version with filtered_cell_indices attribute
    total_cells = sum(len(bpcells.cpp.cell_names_fragments_dir(f)) for f in fragments_normalized)
    
    # Handle pd.Categorical input
    if isinstance(cell_groups, pd.Categorical):
        # Check if this is a filtered categorical (has filtered_cell_indices attribute)
        if hasattr(cell_groups, 'filtered_cell_indices'):
            # Filtered version: create full array with -1 for excluded cells
            filtered_indices = cell_groups.filtered_cell_indices
            cell_groups_array = np.full(total_cells, -1, dtype=np.int32)
            # Map filtered cells to their group codes
            cell_groups_array[filtered_indices] = cell_groups.codes.astype(np.int32)
            
            # Extract group names from filtered cells only, preserving order from original group_order
            # This ensures consistency: group_names matches the actual groups in the matrix
            if group_names is None:
                # Use original_group_order if available (from build_cell_groups), otherwise use categories
                original_order = getattr(cell_groups, 'original_group_order', None)
                group_names = _extract_valid_group_names(cell_groups, original_order)
        else:
            # Full version: all cells included
            # Extract group names from categorical only if not provided by user
            if group_names is None:
                # Use original_group_order if available, otherwise use categories
                original_order = getattr(cell_groups, 'original_group_order', None)
                if original_order is not None:
                    # Only include categories that actually have cells
                    group_names = _extract_valid_group_names(cell_groups, original_order)
                else:
                    # Fall back to all categories (backward compatibility)
                    group_names = list(cell_groups.categories)
            
            if len(cell_groups) != total_cells:
                if len(fragments_normalized) > 1:
                    raise ValueError(
                        f"cell_groups length ({len(cell_groups)}) does not match total cells across fragments ({total_cells}). "
                        f"When using multiple fragments, ensure cell_groups was created using build_cell_groups() "
                        f"with the dict-based API (cell_ids and group_ids as dicts)."
                    )
                else:
                    raise ValueError(
                        f"cell_groups length ({len(cell_groups)}) does not match number of cells in fragment ({total_cells})."
                    )
            # Convert to integer array
            cell_groups_array = cell_groups.codes.astype(np.int32)
            cell_groups_array[cell_groups_array == -1] = -1  # Ensure NaN becomes -1
    else:
        # Non-categorical input: validate length
        if len(cell_groups) != total_cells:
            if len(fragments_normalized) > 1:
                raise ValueError(
                    f"cell_groups length ({len(cell_groups)}) does not match total cells across fragments ({total_cells}). "
                    f"When using multiple fragments, ensure cell_groups was created using build_cell_groups() "
                    f"with the dict-based API (cell_ids and group_ids as dicts)."
                )
            else:
                raise ValueError(
                    f"cell_groups length ({len(cell_groups)}) does not match number of cells in fragment ({total_cells})."
                )
        cell_groups_array = cell_groups

    if isinstance(chrom_sizes, str):
        chrom_sizes = pd.read_csv(chrom_sizes, sep="\t", names=["chrom", "size"])
        chrom_sizes = {t.chrom: t.size for t in chrom_sizes.itertuples()}

    # Re-order chrom_sizes to match the fragment file chromosome order (use first file as reference)
    chrom_order = bpcells.cpp.chr_names_fragments_dir(fragments_normalized[0])
    chrom_sizes = dict(i for i in chrom_sizes.items() if i[0] in chrom_order)
    chrom_sizes = dict(sorted(chrom_sizes.items(), key = lambda x: chrom_order.index(x[0])))

    # Use context manager to ensure temp directory stays alive during C++ execution
    # and is properly cleaned up even if an exception occurs
    with tempfile.TemporaryDirectory() as tmp_dir:
        bpcells.cpp.precalculate_pseudobulk_coverage(
            fragments_normalized,
            output_dir,
            tmp_dir,
            list(chrom_sizes.keys()),
            list(chrom_sizes.values()),
            cell_groups_array,
            1,
            threads,
            group_names
        )

    # Filter library_size.json if we had filtered cells
    # The C++ code may write library sizes for all groups, but we only want filtered ones
    # IMPORTANT: group_names should already be filtered to only include groups with cells,
    # so we need to ensure library_size matches group_names length
    if isinstance(cell_groups, pd.Categorical) and hasattr(cell_groups, 'filtered_cell_indices'):
        library_size_path = os.path.join(output_dir, "library_size.json")
        if os.path.exists(library_size_path):
            try:
                with open(library_size_path, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "library_sizes" in data:
                        all_library_sizes = data["library_sizes"]
                        
                        # Map group_names to their codes in the categorical
                        # The C++ code writes library sizes in the order of group_names passed to it,
                        # which should match the order of categories in the categorical
                        # But we need to ensure we only keep library sizes for groups that have cells
                        category_to_code = {cat: code for code, cat in enumerate(cell_groups.categories)}
                        
                        # Get unique codes for groups that actually have cells (non-NaN codes)
                        unique_codes = sorted(set(cell_groups.codes[cell_groups.codes >= 0]))
                        
                        # Extract library sizes for these codes
                        # The C++ code writes library sizes in the order of group_names we pass,
                        # which should match the categorical categories order
                        filtered_library_sizes = []
                        for code in unique_codes:
                            if code >= 0 and code < len(all_library_sizes):
                                filtered_library_sizes.append(all_library_sizes[code])
                        
                        # Validate: library_size length should match group_names length
                        if len(filtered_library_sizes) != len(group_names):
                            import warnings
                            warnings.warn(
                                f"Library size count ({len(filtered_library_sizes)}) does not match "
                                f"group_names count ({len(group_names)}). This may indicate a mismatch "
                                f"between categorical categories and group_names. "
                                f"Using library sizes for {len(unique_codes)} groups.",
                                UserWarning,
                                stacklevel=2
                            )
                            # Use the filtered library sizes anyway (better than nothing)
                            data["library_sizes"] = filtered_library_sizes
                        else:
                            # Only rewrite if we filtered out some library sizes
                            if len(filtered_library_sizes) < len(all_library_sizes):
                                data["library_sizes"] = filtered_library_sizes
                        
                        # Write the filtered library sizes
                        with open(library_size_path, "w") as f:
                            json.dump(data, f, indent=2)
            except Exception as e:
                import warnings
                warnings.warn(
                    f"Could not filter library sizes: {e}",
                    UserWarning,
                    stacklevel=2
                )

    chrom_offsets = dict(zip(chrom_sizes.keys(), [0] + np.cumsum(list(chrom_sizes.values()))[:-1].tolist()))
    json.dump(chrom_offsets, open(f"{output_dir}/chrom_offsets.json", "w"), indent=2)
    
    # Save group_names.json to ensure consistency and easy access
    # This helps downstream code verify that group_names, library_size, and matrix shape match
    if group_names is not None:
        group_names_path = os.path.join(output_dir, "group_names.json")
        with open(group_names_path, "w") as f:
            json.dump(group_names, f, indent=2)
    
    # Validate consistency: matrix shape should match group_names length
    try:
        matrix = PrecalculatedInsertionMatrix(output_dir)
        if group_names is not None and matrix.shape[0] != len(group_names):
            import warnings
            warnings.warn(
                f"Inconsistency detected: matrix has {matrix.shape[0]} rows but group_names has "
                f"{len(group_names)} entries. This may indicate an issue with group filtering. "
                f"Matrix shape: {matrix.shape}, group_names: {group_names[:5]}... (showing first 5)",
                UserWarning,
                stacklevel=2
            )
        # Also validate library_size length
        if hasattr(matrix, 'library_size') and matrix.library_size is not None:
            if len(matrix.library_size) != matrix.shape[0]:
                import warnings
                warnings.warn(
                    f"Inconsistency detected: library_size has {len(matrix.library_size)} entries "
                    f"but matrix has {matrix.shape[0]} rows. This may indicate an issue with "
                    f"library size filtering.",
                    UserWarning,
                    stacklevel=2
                )
    except Exception as e:
        # Don't fail if validation fails, just warn
        import warnings
        warnings.warn(
            f"Could not validate matrix consistency: {e}",
            UserWarning,
            stacklevel=2
        )
    
    return PrecalculatedInsertionMatrix(output_dir)


def precalculate_insertion_counts_binned(
    fragments: Union[str, List[str]], 
    output_dir: str, 
    cell_groups: Union[Sequence[int], pd.Categorical],
    chrom_sizes: Union[str, Dict[str, int]], 
    bin_size: int = 500,
    threads: int = 16,
    group_names: Optional[List[str]] = None
) -> 'DirMatrix':
    """Precalculate binned insertion counts from fragment data
    
    This function creates a binned genome-wide insertion count matrix where each column
    represents a genomic bin of size `bin_size` base pairs. The result is stored in
    BPCells matrix format and can be loaded as a scipy sparse matrix using DirMatrix.
    
    The current implementation is EXPERIMENTAL, and will crash for matrices with more than
    2^32-1 non-zero entries.

    Args:
        fragments (str | list[str]): Path to a BPCells fragments directory, or list of paths to multiple fragment directories
        output_dir (str): Path to save the insertion counts in
        cell_groups (list[int] or pd.Categorical): Pseudobulk groupings as created by :func:`build_cell_groups()`.
            Should be the output of :func:`build_cell_groups()` to ensure correct cell-to-fragment mapping.
            When using multiple fragment files, the cell_groups should index across all files combined
            (e.g., if file1 has 100 cells and file2 has 150 cells, cell_groups should have length 250).
            If pd.Categorical, group names are taken from the categories.
        chrom_sizes (str | dict[str, int]): Path/URL of UCSC-style chrom.sizes file, or dictionary mapping chromosome names to sizes
        bin_size (int): Size of each genomic bin in base pairs. Default is 500bp.
            The genome will be divided into bins of this size, with each column representing one bin.
            The bin_size does NOT need to equal the genome size - it's used to bin the genome.
        threads (int): Number of threads to use during matrix calculation (default = 16, creates threads*4 chunks for parallelization)
        group_names (list[str], optional): Names for each group in the same order as group indices (0, 1, 2, ...).
            Ignored if cell_groups is pd.Categorical.

    Returns:
        DirMatrix: A disk-backed BPCells matrix object. The matrix has shape (n_pseudobulks, n_bins) where:
            - n_pseudobulks = number of cell groups
            - n_bins = total number of bins across all chromosomes (sum of ceil(chr_len / bin_size) for each chromosome)
        
        You can slice the DirMatrix to get scipy sparse matrices:
        
        >>> mat = precalculate_insertion_counts_binned(...)
        >>> # Get all data as scipy sparse matrix
        >>> sparse_mat = mat[:, :]  # Returns scipy.sparse.csc_matrix
        >>> # Get specific rows (pseudobulks) and columns (bins)
        >>> subset = mat[0:5, 100:200]  # Returns scipy.sparse.csc_matrix

    Examples
    --------
    
    Create a 500bp binned matrix:
    
    >>> cell_groups = build_cell_groups(...)
    >>> mat = precalculate_insertion_counts_binned(
    ...     fragments="/path/to/fragments",
    ...     output_dir="/path/to/output",
    ...     cell_groups=cell_groups,
    ...     chrom_sizes={"chr1": 248956422, "chr2": 242193529},
    ...     bin_size=500,
    ...     threads=8
    ... )
    >>> # Access as scipy sparse matrix
    >>> sparse_mat = mat[:, :]
    >>> print(f"Matrix shape: {sparse_mat.shape}")  # (n_pseudobulks, n_bins)
    
    See Also:
        :func:`precalculate_insertion_counts` : Per-base (1bp) precalculation
        :func:`build_cell_groups` : Create cell_groups categorical from fragments
        :class:`DirMatrix` : Disk-backed matrix interface
    """
    if DirMatrix is None:
        raise ImportError(
            "DirMatrix is not available. Please ensure bpcells.experimental.matrix is importable."
        )
    
    if bin_size < 1:
        raise ValueError(f"bin_size must be >= 1, got {bin_size}")
    
    # Convert single path to list for uniform handling
    if isinstance(fragments, str):
        fragments = [fragments]
    
    # Normalize fragment paths to absolute paths for consistency
    fragments_normalized = [os.path.abspath(os.path.expanduser(f)) for f in fragments]
    
    # Validate that cell_groups length matches total cells across fragments
    # OR that it's a filtered version with filtered_cell_indices attribute
    total_cells = sum(len(bpcells.cpp.cell_names_fragments_dir(f)) for f in fragments_normalized)
    
    # Handle pd.Categorical input (same logic as precalculate_insertion_counts)
    if isinstance(cell_groups, pd.Categorical):
        # Check if this is a filtered categorical (has filtered_cell_indices attribute)
        if hasattr(cell_groups, 'filtered_cell_indices'):
            # Filtered version: create full array with -1 for excluded cells
            filtered_indices = cell_groups.filtered_cell_indices
            cell_groups_array = np.full(total_cells, -1, dtype=np.int32)
            # Map filtered cells to their group codes
            cell_groups_array[filtered_indices] = cell_groups.codes.astype(np.int32)
            
            # Extract group names from filtered cells only
            if group_names is None:
                original_order = getattr(cell_groups, 'original_group_order', None)
                group_names = _extract_valid_group_names(cell_groups, original_order)
        else:
            # Full version: all cells included
            if group_names is None:
                original_order = getattr(cell_groups, 'original_group_order', None)
                if original_order is not None:
                    group_names = _extract_valid_group_names(cell_groups, original_order)
                else:
                    group_names = list(cell_groups.categories)
            
            if len(cell_groups) != total_cells:
                if len(fragments_normalized) > 1:
                    raise ValueError(
                        f"cell_groups length ({len(cell_groups)}) does not match total cells across fragments ({total_cells}). "
                        f"When using multiple fragments, ensure cell_groups was created using build_cell_groups() "
                        f"with the dict-based API (cell_ids and group_ids as dicts)."
                    )
                else:
                    raise ValueError(
                        f"cell_groups length ({len(cell_groups)}) does not match number of cells in fragment ({total_cells})."
                    )
            cell_groups_array = cell_groups.codes.astype(np.int32)
            cell_groups_array[cell_groups_array == -1] = -1
    else:
        # Non-categorical input: validate length
        if len(cell_groups) != total_cells:
            if len(fragments_normalized) > 1:
                raise ValueError(
                    f"cell_groups length ({len(cell_groups)}) does not match total cells across fragments ({total_cells}). "
                    f"When using multiple fragments, ensure cell_groups was created using build_cell_groups() "
                    f"with the dict-based API (cell_ids and group_ids as dicts)."
                )
            else:
                raise ValueError(
                    f"cell_groups length ({len(cell_groups)}) does not match number of cells in fragment ({total_cells})."
                )
        cell_groups_array = cell_groups

    if isinstance(chrom_sizes, str):
        chrom_sizes = pd.read_csv(chrom_sizes, sep="\t", names=["chrom", "size"])
        chrom_sizes = {t.chrom: t.size for t in chrom_sizes.itertuples()}

    # Re-order chrom_sizes to match the fragment file chromosome order (use first file as reference)
    chrom_order = bpcells.cpp.chr_names_fragments_dir(fragments_normalized[0])
    chrom_sizes = dict(i for i in chrom_sizes.items() if i[0] in chrom_order)
    chrom_sizes = dict(sorted(chrom_sizes.items(), key = lambda x: chrom_order.index(x[0])))

    # Calculate chromosome lengths in bins (for metadata)
    chrom_bin_counts = {chr: (size + bin_size - 1) // bin_size for chr, size in chrom_sizes.items()}
    total_bins = sum(chrom_bin_counts.values())
    
    # Use context manager to ensure temp directory stays alive during C++ execution
    # and is properly cleaned up even if an exception occurs
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Call C++ function with specified bin_size
        # Note: tmp_dir will be cleaned up automatically when this block exits
        bpcells.cpp.precalculate_pseudobulk_coverage(
            fragments_normalized,
            output_dir,
            tmp_dir,
            list(chrom_sizes.keys()),
            list(chrom_sizes.values()),
            cell_groups_array,
            bin_size,  # Use the specified bin_size instead of hardcoded 1
            threads,
            group_names
        )
    
    # Save metadata about binning
    metadata = {
        "bin_size": bin_size,
        "chrom_sizes": chrom_sizes,
        "chrom_bin_counts": chrom_bin_counts,
        "total_bins": total_bins,
        "group_names": group_names if group_names is not None else []
    }
    metadata_path = os.path.join(output_dir, "binning_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save group_names.json for consistency
    if group_names is not None:
        group_names_path = os.path.join(output_dir, "group_names.json")
        with open(group_names_path, "w") as f:
            json.dump(group_names, f, indent=2)
    
    # Return DirMatrix object (can be sliced to get scipy sparse matrices)
    return DirMatrix(output_dir)

