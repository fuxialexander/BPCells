# Copyright 2023 BPCells contributors
# 
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

import bpcells
import bpcells.cpp

import copy
import os.path
from typing import List, Optional

import numpy as np
import scipy

from ._slicing import normalize_index

class DirMatrix:
    """
    Disk-backed BPCells integer matrix

    This reads BPCells-format matrices, returning scipy.sparse.csc_matrix objects when sliced.

    Args:
        dir (str): Path of the matrix directory
    
    Examples
    --------

    >>> from bpcells import DirMatrix
    >>> mat = DirMatrix("/path/to/matrix")
    >>> mat[:,[1,3,2,4]]
    <3x4 sparse matrix of type '<class 'numpy.uint32'>'
        with 6 stored elements in Compressed Sparse Column format>
    """
    def __init__(self, dir: str):
        self.dir = str(os.path.abspath(os.path.expanduser(dir)))
        self._transpose = open(os.path.join(self.dir, "storage_order"), "r").readline().strip() == "row"
        self.shape = bpcells.cpp.dims_matrix_dir(self.dir)
        """Tuple[int,int]: Dimensions of matrix"""
        if self._transpose:
            self.shape = (self.shape[1], self.shape[0])
        
        self.dtype = np.uint32
        self.threads = 0
        """int: Number of threads to use for reading (default=1)"""
    
    def __repr__(self):
        orientation = "row-major" if self._transpose else "col-major"
        return f"<{self.shape[0]}x{self.shape[1]} {orientation} sparse array stored in \n\t{self.dir}>"
    
    def transpose(self) -> 'DirMatrix':
        """Return at transposed view of the matrix"""
        x = copy.copy(self)
        x._transpose = not x._transpose
        x.shape = (x.shape[1], x.shape[0])
        return x

    @property
    def T(self) -> 'DirMatrix':
        """Return a transposed view of the matrix"""
        return self.transpose()
    
    def __getitem__(self, key) -> scipy.sparse.spmatrix:
        row, col = normalize_index(key, self.shape)

        if self._transpose:
            row, col = col, row

        if isinstance(col, slice):
            col = list(range(col.start, col.stop, col.step))

        n_rows = self.shape[1 if self._transpose else 0]
        all_rows = slice(0, n_rows, 1)
        if isinstance(row, slice):
            if row != all_rows:
                row = list(range(row.start, row.stop, row.step))
            else:
                row = None
        
        res = scipy.sparse.hstack(bpcells.cpp.load_matrix_dir_subset(self.dir, row, col, self.threads))
        
        if self._transpose:
            return res.T
        else:
            return res
    
    @staticmethod
    def calculate_safe_chunk_size(rows: int, max_chunk_size: int = 10000) -> int:
        """Calculate safe chunk size for column slicing to avoid uint32_t overflow.
        
        When loading large matrices with many rows, slicing too many columns at once
        can cause uint32_t overflow in the C++ backend, leading to segfaults.
        This function calculates the maximum safe number of columns to load at once.
        
        Args:
            rows: Number of rows in the matrix
            max_chunk_size: Maximum desired chunk size (default: 10000).
                The actual chunk size will be min(max_chunk_size, safe_limit).
        
        Returns:
            Safe chunk size (number of columns) that won't cause uint32_t overflow.
            The calculation ensures rows * chunk_size <= UINT32_MAX * 0.9 to account
            for internal C++ calculations.
        
        Examples:
            >>> mat = DirMatrix("/path/to/matrix")
            >>> rows, cols = mat.shape
            >>> chunk_size = DirMatrix.calculate_safe_chunk_size(rows)
            >>> # Load in chunks
            >>> for start_col in range(0, cols, chunk_size):
            ...     chunk = mat[:, start_col:start_col + chunk_size]
        """
        UINT32_MAX = 2**32 - 1
        # Use 90% of max to be safe (account for any internal calculations)
        max_safe_chunk_size = int((UINT32_MAX * 0.9) // rows)
        return min(max_chunk_size, max_safe_chunk_size)
    
    def load_chunked_columns(
        self, 
        start_col: int = 0, 
        end_col: Optional[int] = None,
        chunk_size: Optional[int] = None,
        format: str = 'csr'
    ) -> List[scipy.sparse.spmatrix]:
        """Load matrix columns in chunks to avoid uint32_t overflow.
        
        This method automatically calculates safe chunk sizes and loads columns
        in chunks, returning a list of sparse matrices that can be concatenated
        with scipy.sparse.hstack().
        
        Args:
            start_col: Starting column index (default: 0)
            end_col: Ending column index (default: all columns)
            chunk_size: Chunk size in columns. If None, automatically calculated
                using calculate_safe_chunk_size().
            format: Sparse matrix format for returned chunks ('csr', 'csc', 'coo').
                Default: 'csr'
        
        Returns:
            List of sparse matrices, one per chunk. Each matrix has shape
            (rows, chunk_cols) where chunk_cols may vary for the last chunk.
        
        Examples:
            >>> mat = DirMatrix("/path/to/large_matrix")
            >>> chunks = mat.load_chunked_columns()
            >>> combined = scipy.sparse.hstack(chunks, format='csr')
        """
        rows, cols = self.shape
        
        if end_col is None:
            end_col = cols
        
        if chunk_size is None:
            chunk_size = self.calculate_safe_chunk_size(rows)
        
        # Validate chunk size won't cause overflow
        UINT32_MAX = 2**32 - 1
        potential_elements = rows * chunk_size
        if potential_elements > UINT32_MAX:
            raise ValueError(
                f"Chunk size {chunk_size} would cause uint32_t overflow: "
                f"rows ({rows:,}) × chunk_size ({chunk_size:,}) = {potential_elements:,} > {UINT32_MAX:,}"
            )
        
        chunks = []
        for start in range(start_col, end_col, chunk_size):
            end = min(start + chunk_size, end_col)
            chunk = self[:, start:end]
            
            # Convert to requested format
            if format == 'csr':
                chunks.append(chunk.tocsr())
            elif format == 'csc':
                chunks.append(chunk.tocsc())
            elif format == 'coo':
                chunks.append(chunk.tocoo())
            else:
                raise ValueError(f"Unsupported format: {format}. Use 'csr', 'csc', or 'coo'")
        
        return chunks
    
    @classmethod
    def from_scipy_sparse(cls, scipy_mat: scipy.sparse.spmatrix, dir: str) -> 'DirMatrix':
        """Create a DirMatrix from a scipy sparse matrix.
        
        Will write in compressed sparse column format for all input types
        other than scipy.sparse.csr_matrix

        Args:
            scipy_mat (scipy.spmatrix): Scipy sparse matrix
            dir (str): Path to write the matrix

        Returns:
            DirMatrix: View of the matrix written to disk
        """

        # Normalize dir
        dir = str(os.path.abspath(os.path.expanduser(dir)))

        # Handle row-major scipy sparse
        row_major = scipy_mat.format == "csr"
        if row_major:
            scipy_mat = scipy_mat.T

        bpcells.cpp.write_matrix_dir_from_memory(scipy_mat, dir, row_major)
        
        return cls(dir)
    
    @classmethod
    def _from_stack_helper(cls, mats, out_dir, is_horizontal):
        if len(mats) == 0:
            raise Exception("Must provide non-zero number of input matrices")
        
        if len(mats) == 1:
            return mats[0]
        
        # Normalize dir
        out_dir = str(os.path.abspath(os.path.expanduser(out_dir)))
        
        if len(set(m._transpose for m in mats)) != 1:
            raise Exception("Not all input matrices have the same on-disk storage order (row-major vs col-major)")

        # For hstack (horizontal/column-wise): need same rows
        # For vstack (vertical/row-wise): need same columns
        if is_horizontal:
            if len(set(m.shape[0] for m in mats)) != 1:
                raise Exception("Not all input matrices have same number of rows (required for hstack)")
        else:
            if len(set(m.shape[1] for m in mats)) != 1:
                raise Exception("Not all input matrices have same number of columns (required for vstack)")
        
        # Validate dimensions to prevent uint32_t overflow
        # uint32_t max value is 2^32 - 1 = 4,294,967,295
        UINT32_MAX = 2**32 - 1
        
        if is_horizontal:
            # For hstack: concatenating columns, so check total columns
            total_cols = sum(m.shape[1] for m in mats)
            if total_cols > UINT32_MAX:
                raise ValueError(
                    f"Total columns ({total_cols:,}) exceeds uint32_t maximum ({UINT32_MAX:,}). "
                    f"This would cause overflow in C++ code. Consider splitting into smaller chunks."
                )
            # Check individual matrix dimensions
            for i, m in enumerate(mats):
                if m.shape[0] > UINT32_MAX or m.shape[1] > UINT32_MAX:
                    raise ValueError(
                        f"Matrix {i} has dimensions ({m.shape[0]:,}, {m.shape[1]:,}) "
                        f"exceeding uint32_t maximum ({UINT32_MAX:,})"
                    )
            # For hstack (ConcatCols), check for potential column offset overflow
            # Column offsets are used for seeking, check if any intermediate sum would overflow
            running_total = 0
            for i, m in enumerate(mats):
                cols = m.shape[1]
                if running_total > UINT32_MAX - cols:
                    raise ValueError(
                        f"Column offset overflow: after matrix {i}, running total ({running_total:,}) + "
                        f"columns ({cols:,}) would exceed uint32_t maximum ({UINT32_MAX:,})"
                    )
                running_total += cols
        else:
            # For vstack: concatenating rows, so check total rows
            total_rows = sum(m.shape[0] for m in mats)
            if total_rows > UINT32_MAX:
                raise ValueError(
                    f"Total rows ({total_rows:,}) exceeds uint32_t maximum ({UINT32_MAX:,}). "
                    f"This would cause overflow in C++ code. Consider splitting into smaller chunks."
                )
            # Check individual matrix dimensions
            for i, m in enumerate(mats):
                if m.shape[0] > UINT32_MAX or m.shape[1] > UINT32_MAX:
                    raise ValueError(
                        f"Matrix {i} has dimensions ({m.shape[0]:,}, {m.shape[1]:,}) "
                        f"exceeding uint32_t maximum ({UINT32_MAX:,})"
                    )
            # For vstack (ConcatRows), check for potential row index overflow
            # Row indices in the data are 0-based (0 to rows-1), and we add row_offset
            # The maximum final row index = (max_rows_in_last_matrix - 1) + row_offset_for_last_matrix
            # = (max_rows_in_last_matrix - 1) + (total_rows - max_rows_in_last_matrix)
            # = total_rows - 1
            # So if total_rows <= UINT32_MAX, we're safe. But we also need to check intermediate
            # calculations don't overflow during offset accumulation.
            # Check if any intermediate sum would overflow
            running_total = 0
            for i, m in enumerate(mats):
                rows = m.shape[0]
                if running_total > UINT32_MAX - rows:
                    raise ValueError(
                        f"Row offset overflow: after matrix {i}, running total ({running_total:,}) + "
                        f"rows ({rows:,}) would exceed uint32_t maximum ({UINT32_MAX:,})"
                    )
                running_total += rows
        
        apply_transpose = mats[0]._transpose

        bpcells.cpp.write_matrix_dir_from_concat([m.dir for m in mats], out_dir, is_horizontal != apply_transpose)

        if apply_transpose:
            return cls(out_dir).T
        else:
            return cls(out_dir)
    
    @classmethod
    def from_hstack(cls, mats: List['DirMatrix'], out_dir: str) -> 'DirMatrix':
        """Create a DirMatrix by concatenating a list of DirMatrix objects horizontally (column wise)

        Args:
            mats (List[DirMatrix]): List of input matrices
            out_dir (str): Output path for DirMatrix

        Returns:
            DirMatrix: View of the matrix written to disk
        """
        return cls._from_stack_helper(mats, out_dir, True)
        
    
    @classmethod
    def from_vstack(cls, mats: List['DirMatrix'], out_dir: str) -> 'DirMatrix':
        """Create a DirMatrix by concatenating a list of DirMatrix objects vertically (row wise)

        Args:
            mats (List[DirMatrix]): List of input matrices
            out_dir (str): Output path for DirMatrix

        Returns:
            DirMatrix: View of the matrix written to disk
        """
        return cls._from_stack_helper(mats, out_dir, False)

    @classmethod
    def from_h5ad(cls, h5ad_path: str, out_dir: str, group: str = "X") -> 'DirMatrix':
        """Create a DirMatrix from an h5ad file. 
        Truncates floating point values to integers

        Args:
            h5ad_path (str): Path to h5ad file
            out_dir (str): Output path for DirMatrix
            group (str, optional): HDF5 group to read matrix from. Defaults to "X".

        Returns:
            DirMatrix: View of the matrix written to disk
        """
        # Normalize paths
        h5ad_path = str(os.path.abspath(os.path.expanduser(h5ad_path)))
        out_dir = str(os.path.abspath(os.path.expanduser(out_dir)))
        bpcells.cpp.write_matrix_dir_from_h5ad(h5ad_path, out_dir, group)
        return cls(out_dir)
        


class MemMatrix:
    """
    In-memory BPCells integer matrix

    This reads BPCells-format matrices from disk, returning scipy.sparse.csc_matrix objects when sliced.
    It is much more memory-intensive, but consistently fast for random reads

    Args:
        dir (str): Path of the matrix directory
    
    Examples
    --------

    >>> from bpcells import MemMatrix
    >>> mat = MemMatrix("/path/to/matrix")
    >>> mat[:,[1,3,2,4]]
    <3x4 sparse matrix of type '<class 'numpy.uint32'>'
        with 6 stored elements in Compressed Sparse Column format>
    """

    def __init__(self, dir: str, threads: int = 0):
        dir = str(os.path.abspath(os.path.expanduser(dir)))
        # Confirm shape, while having a side-effect of validating the matrix
        self.shape = bpcells.cpp.dims_matrix_dir(dir)
        """Dimensions of matrix"""

        self._transpose = open(os.path.join(dir, "storage_order"), "r").read() == "row"
        if self._transpose:
            self.shape[0], self.shape[1] = self.shape[1], self.shape[0]
        
        self.dtype = np.uint32
        self.threads = 0
        """Threads used for reads (default=1)"""

        self._data = bpcells.cpp.load_matrix_dir_to_memory(dir)
        
    def transpose(self) -> 'MemMatrix':
        """Return at transposed view of the matrix"""
        x = copy.copy(self)
        x._transpose = not x._transpose
        x.shape = (x.shape[1], x.shape[0])
        return x

    @property
    def T(self) -> 'MemMatrix':
        """Return a transposed view of the matrix"""
        return self.transpose()
        
    def __repr__(self):
        orientation = "row-major" if self._transpose else "col-major"
        return f"<{self.shape[0]}x{self.shape[1]} {orientation} sparse array stored in memory>"
    
    def __getitem__(self, key) -> scipy.sparse.spmatrix:
        row, col = normalize_index(key, self.shape)

        if self._transpose:
            row, col = col, row

        if isinstance(col, slice):
            col = list(range(col.start, col.stop, col.step))

        n_rows = self.shape[1 if self._transpose else 0]
        all_rows = slice(0, n_rows, 1)
        if isinstance(row, slice):
            if row != all_rows:
                row = list(range(row.start, row.stop, row.step))
            else:
                row = None
        
        res = scipy.sparse.hstack(bpcells.cpp.load_matrix_memory_subset(self._data, row, col, self.threads))
        
        if self._transpose:
            return res.T
        else:
            return res