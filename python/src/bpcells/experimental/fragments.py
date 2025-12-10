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
    group_order: Sequence[str]
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

    Returns:
        pd.Categorical:
        Pandas Categorical suitable as input for ``cell_groups`` in :func:`pseudobulk_insertion_counts()`.
        Same length as total number of cells in the ``fragments`` input, specifying the output
        pseudobulk group for each cell (or NaN if the cell is excluded from consideration).
        When using multiple fragment files, the categorical covers all cells across all files.
        The categories are ordered according to ``group_order``.

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
    # Convert single path to list for uniform handling
    if isinstance(fragments, str):
        fragments = [fragments]
    
    # Normalize fragment paths to absolute paths for consistent matching
    fragments_normalized = [os.path.abspath(os.path.expanduser(f)) for f in fragments]

    # Determine if using dict-based API (for multiple fragments) or list-based API
    using_dict_api = isinstance(cell_ids, dict)
    
    # Warn if list-based API is used with multiple fragments
    if not using_dict_api and len(fragments_normalized) > 1:
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
            raise ValueError(f"Missing cell_ids entries for fragments: {missing_frags}")
        missing_frags = set(fragments_normalized) - set(group_ids_normalized.keys())
        if missing_frags:
            raise ValueError(f"Missing group_ids entries for fragments: {missing_frags}")
        
        # Validate lengths match for each fragment
        for frag_path in fragments_normalized:
            if len(cell_ids_normalized[frag_path]) != len(group_ids_normalized[frag_path]):
                raise ValueError(
                    f"cell_ids and group_ids must have same length for fragment {frag_path}. "
                    f"Got {len(cell_ids_normalized[frag_path])} and {len(group_ids_normalized[frag_path])}"
                )
    else:
        # List-based API: cell_ids and group_ids are sequences
        if isinstance(group_ids, dict):
            raise TypeError("When cell_ids is a list, group_ids must also be a list")
        if len(cell_ids) != len(group_ids):
            raise ValueError(f"cell_ids and group_ids must have same length. Got {len(cell_ids)} and {len(group_ids)}")

    # Build cell index lookup across all fragment files
    # Track cells in order across fragments to preserve fragment context for duplicate barcodes
    # Sequential matching is O(total_cells) which is optimal since we iterate through all cells anyway
    cell_sequence = []  # List of (frag_path, cell_name, global_index) in order across all fragments
    current_index = 0
    for frag_path in fragments_normalized:
        for cell_name in bpcells.cpp.cell_names_fragments_dir(frag_path):
            cell_sequence.append((frag_path, cell_name, current_index))
            current_index += 1

    # Total number of cells across all fragments
    total_cells = current_index

    # Create array of group assignments
    cell_groups = [None] * total_cells

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
                raise ValueError(f"group_ids contains groups not in group_order: {missing}")
        
        # Match cells using fragment-specific lookup
        for frag_path, cell_name, global_idx in cell_sequence:
            if frag_path in frag_cell_maps and cell_name in frag_cell_maps[frag_path]:
                # Match found - assign the corresponding group_id
                cell_groups[global_idx] = frag_cell_maps[frag_path][cell_name]
            # If no match, cell_groups[global_idx] remains None (cell excluded)
    else:
        # List-based API: sequential matching (original behavior)
        # Validate that all group_ids are in group_order
        unique_groups = set(group_ids)
        if not unique_groups <= set(group_order):
            missing = unique_groups - set(group_order)
            raise ValueError(f"group_ids contains groups not in group_order: {missing}")
        
        # Match cell_ids sequentially with cell_sequence
        # This is O(total_cells) which is optimal - we iterate through all cells once
        # Duplicate barcodes from different fragments are handled correctly because we match in order
        # cell_ids should be in the same order as cells appear across fragments
        cell_id_idx = 0
        for frag_path, cell_name, global_idx in cell_sequence:
            if cell_id_idx < len(cell_ids) and cell_ids[cell_id_idx] == cell_name:
                # Match found - assign the corresponding group_id
                # This handles duplicate barcodes correctly because we match in order
                cell_groups[global_idx] = group_ids[cell_id_idx]
                cell_id_idx += 1
            # If no match, cell_groups[global_idx] remains None (cell excluded)

    # Create categorical with ordered categories
    return pd.Categorical(cell_groups, categories=group_order, ordered=True)

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
    # This helps catch errors early if cell_groups was created incorrectly
    total_cells = sum(len(bpcells.cpp.cell_names_fragments_dir(f)) for f in fragments_normalized)
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

    # Handle pd.Categorical input
    if isinstance(cell_groups, pd.Categorical):
        # Extract group names from categorical only if not provided by user
        if group_names is None:
            group_names = list(cell_groups.categories)
        # Convert to integer array
        cell_groups_array = cell_groups.codes.astype(np.int32)
        cell_groups_array[cell_groups_array == -1] = -1  # Ensure NaN becomes -1
    else:
        cell_groups_array = cell_groups

    if isinstance(chrom_sizes, str):
        chrom_sizes = pd.read_csv(chrom_sizes, sep="\t", names=["chrom", "size"])
        chrom_sizes = {t.chrom: t.size for t in chrom_sizes.itertuples()}

    # Re-order chrom_sizes to match the fragment file chromosome order (use first file as reference)
    chrom_order = bpcells.cpp.chr_names_fragments_dir(fragments_normalized[0])
    chrom_sizes = dict(i for i in chrom_sizes.items() if i[0] in chrom_order)
    chrom_sizes = dict(sorted(chrom_sizes.items(), key = lambda x: chrom_order.index(x[0])))

    tmp = tempfile.TemporaryDirectory()
    bpcells.cpp.precalculate_pseudobulk_coverage(
        fragments_normalized,
        output_dir,
        tmp.name,
        list(chrom_sizes.keys()),
        list(chrom_sizes.values()),
        cell_groups_array,
        1,
        threads,
        group_names
    )

    chrom_offsets = dict(zip(chrom_sizes.keys(), [0] + np.cumsum(list(chrom_sizes.values()))[:-1].tolist()))
    json.dump(chrom_offsets, open(f"{output_dir}/chrom_offsets.json", "w"), indent=2)
    return PrecalculatedInsertionMatrix(output_dir)

