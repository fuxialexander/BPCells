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

def build_cell_groups(fragments: str, cell_ids: Sequence[str], group_ids: Sequence[str], group_order: Sequence[str]) -> pd.Categorical:
    """Build cell_groups categorical for use in :func:`pseudobulk_insertion_counts()`

    Args:
        fragments (str): Path to BPCells fragments directory
        cell_ids (list[str]): List of cell IDs
        group_ids (list[str]): List of pseudobulk IDs for each cell (same length as ``cell_ids``)
        group_order (list[str]): Output order of pseudobulks (Contain the unique ``group_ids``)
    
    Returns:
        pd.Categorical: 
        Pandas Categorical suitable as input for ``cell_groups`` in :func:`pseudobulk_insertion_counts()`.
        Same length as total number of cells in the ``fragments`` input, specifying the output
        pseudobulk group for each cell (or NaN if the cell is excluded from consideration).
        The categories are ordered according to ``group_order``.
        
    See Also:
        :func:`pseudobulk_insertion_counts`
    """
    cell_index_lookup = {c: i for i, c in enumerate(bpcells.cpp.cell_names_fragments_dir(fragments))}
    
    assert len(cell_ids) == len(group_ids)
    assert set(group_ids) <= set(group_order)
    
    # Create array of group assignments
    all_cells = bpcells.cpp.cell_names_fragments_dir(fragments)
    cell_groups = [None] * len(all_cells)
    
    for cell_id, group_id in zip(cell_ids, group_ids):
        if cell_id in cell_index_lookup:
            cell_groups[cell_index_lookup[cell_id]] = group_id
    
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

def precalculate_insertion_counts(fragments: str, output_dir: str, cell_groups: Union[Sequence[int], pd.Categorical], 
                                 chrom_sizes: Union[str, Dict[str, int]], threads: int = 0,
                                 group_names: Optional[List[str]] = None):
    """Precalculate per-base insertion counts from fragment data

    The current implementation is EXPERIMENTAL, and will crash for matrices with more than
    2^32-1 non-zero entries.

    Args:
        fragments (str): Path to a BPCells fragments directory
        output_dir (str): Path to save the insertion counts in
        cell_groups (list[int] or pd.Categorical): List of pseudbulk groupings as created by :func:`build_cell_groups()`.
            If pd.Categorical, group names are taken from the categories.
        chrom_sizes (str | dict[str, int]): Path/URL of UCSC-style chrom.sizes file, or dictionary mapping chromosome names to sizes
        threads (int): Number of threads to use during matrix calculation (default = 1)
        group_names (list[str], optional): Names for each group in the same order as group indices (0, 1, 2, ...).
            Ignored if cell_groups is pd.Categorical.
    
    Returns:
        A :class:`PrecalculatedInsertionMatrix` object

    See Also:
        :class:`PrecalculatedInsertionMatrix`
    """
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
    
    # Re-order chrom_sizes to match the fragment file chromosome order
    chrom_order = bpcells.cpp.chr_names_fragments_dir(fragments)
    chrom_sizes = dict(i for i in chrom_sizes.items() if i[0] in chrom_order)
    chrom_sizes = dict(sorted(chrom_sizes.items(), key = lambda x: chrom_order.index(x[0])))

    tmp = tempfile.TemporaryDirectory()
    bpcells.cpp.precalculate_pseudobulk_coverage(
        fragments,
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

