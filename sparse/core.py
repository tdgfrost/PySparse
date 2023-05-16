import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm
import os
from numba import njit, types
from .array_api import SparseArray


"""
============================================================
Sparse array conversion functions
============================================================
"""


def announce_progress(message: str) -> None:
    """
    Print a message to the console
    :param message: message to print
    :return: None
    """

    print('\n' + '='*50 + '\n' + message + '\n' + '='*50)
    return


def __calc_sparse_shape(array: np.ndarray, chunksize: int, verbose: bool) -> tuple:
    """
    Calculate the shape of the (pending) sparse array
    :param array: dense numpy array
    :param chunksize: chunksize to use for calculation - if None, will use the whole array
    :param verbose: whether to print progress statements
    :return: tuple of shape
    """

    data_shape = 0
    shape = array.shape
    announce_progress('Identifying sparse shape...') if verbose else None

    if chunksize is None:
        data_shape = np.count_nonzero(array)

    else:
        for i in tqdm(range(0, shape[0], chunksize)) if verbose else range(0, shape[0], chunksize):
            data_shape += np.count_nonzero(array[i:i + chunksize])

    return (data_shape,)


def __convert_to_sparse_data(array_chunk: np.ndarray, iteration: int) -> (np.ndarray, np.ndarray):
    """
    Convert a chunk of a dense array to sparse data
    :param array_chunk: chunk of dense array
    :param iteration: chunk index
    :return: tuple of sparse coordinates and sparse values
    """

    sparse_coords = np.nonzero(array_chunk)
    sparse_values = array_chunk[sparse_coords]
    sparse_coords = list(sparse_coords)
    sparse_coords[0] += iteration
    sparse_coords = np.stack(sparse_coords, axis=1)

    return sparse_coords, sparse_values


def __write_sparse_arrays(array: np.ndarray, path: 'str', chunksize: int, verbose: bool) -> None:
    """
    Simultaneously convert and write a dense array to sparse arrays
    :param array: dense numpy array to be converted
    :param path: path to write sparse arrays to
    :param chunksize: chunksize to use for conversion - if None, will convert the whole array in memory
    :param verbose: whether to print progress statements
    :return:
    """

    # Identify the relevant shapes of the dense and sparse arrays
    dense_shape = array.shape
    dense_dtype = array.dtype
    nonzero_shape = __calc_sparse_shape(array, chunksize, verbose)

    # Create the sparse array binaries (memory-mapped)
    memmap_sparse_data = open_memmap(os.path.join(path, 'sparse_data.npy'),
                                     dtype=dense_dtype,
                                     mode='w+',
                                     shape=nonzero_shape)

    memmap_sparse_coords = open_memmap(os.path.join(path, 'sparse_coords.npy'),
                                       dtype=np.int32,
                                       mode='w+',
                                       shape=(nonzero_shape[0], len(dense_shape)))

    np.save(os.path.join(path, 'dense_shape.npy'), dense_shape)

    # Convert the dense array to sparse arrays
    announce_progress('Writing sparse arrays...') if verbose else None
    sparse_index = 0

    if chunksize is None:
        sparse_coords, sparse_values = __convert_to_sparse_data(array, 0)

        memmap_sparse_coords[sparse_index:sparse_index + sparse_coords.shape[0]] = sparse_coords
        memmap_sparse_data[sparse_index:sparse_index + sparse_coords.shape[0]] = sparse_values

        sparse_index += sparse_coords.shape[0]

    else:
        for chunk_idx in tqdm(range(0, dense_shape[0], chunksize)) if verbose else range(0, dense_shape[0], chunksize):
            sparse_coords, sparse_values = __convert_to_sparse_data(array[chunk_idx:chunk_idx + chunksize], chunk_idx)

            memmap_sparse_coords[sparse_index:sparse_index + sparse_coords.shape[0]] = sparse_coords
            memmap_sparse_data[sparse_index:sparse_index + sparse_coords.shape[0]] = sparse_values

            sparse_index += sparse_coords.shape[0]

    return


def to_sparse(array: np.ndarray, savepath: 'str', chunksize=1000, verbose=True) -> None:
    """
    Convert and write a dense array to a sparse array
    :param array: numpy array to be converted
    :param savepath: filepath to write sparse array to
    :param chunksize: number of rows to process at a time - if None, will process the whole array in memory
    :param verbose: whether to print progress statements
    :return: None
    """
    if not os.path.isdir(savepath):
        os.makedirs(savepath)

    __write_sparse_arrays(array, savepath, chunksize, verbose)
    return


@njit(types.UniTuple(types.int64, 2)(types.Array(types.int32, 2, 'C', readonly=True), types.int64, types.int64,
                                     types.int64), cache=True)
def find_indices(coords, row_idx, start, end):
    """
    Find the start and end indices of a particular row coordinate in a sparse coordinates array
    :param coords: array of sparse coordinates
    :param row_idx: row index to find
    :param start: start index of the array to search from
    :param end: end index of the array to search to
    :return:
    """
    if start > end:
        # Target number not found in the array
        return -1, -1

    mid = (start + end) // 2

    if coords[mid, 0] < row_idx:
        # Target number is in the right half of the array
        return find_indices(coords, row_idx, mid + 1, end)
    elif coords[mid, 0] > row_idx:
        # Target number is in the left half of the array
        return find_indices(coords, row_idx, start, mid - 1)
    else:
        # Found the target number, now find the start and end indices
        start_index = end_index = mid

        # Find the start index
        step = 1
        while start_index > start and coords[start_index - step, 0] == row_idx:
            start_index -= step
            step *= 2

        # Refine the start index using binary search
        left = start_index - step
        right = start_index
        while left < right:
            mid = (left + right) // 2
            if coords[mid, 0] == row_idx:
                start_index = mid
                right = mid
            else:
                left = mid + 1

        # Find the end index
        step = 1
        while end_index < end and coords[end_index + step, 0] == row_idx:
            end_index += step
            step *= 2

        # Refine the end index using binary search
        left = end_index
        right = end_index + step
        while left < right:
            mid = (left + right) // 2
            if coords[mid, 0] == row_idx:
                end_index = mid
                left = mid + 1
            else:
                right = mid

        return start_index, end_index


def load_sparse(data_path: str, coords_path: str, shape: str or tuple, mode='r'):
    """
    Load a (memory-mapped) sparse array from disk
    :param data_path: path to sparse data array
    :param coords_path: path to sparse coordinates array
    :param shape: shape of the dense array, as either tuple or path to numpy array containing shape
    :param mode: mode to open the sparse arrays in (e.g., read-only, read-write, etc.) - r/r+/w
    :return: SparseArray object
    """
    return SparseArray(data_path, coords_path, shape, mode)

