import numpy as np
from numpy.lib.format import open_memmap
import os
from numba import njit, prange
import pickle

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

    print('\n' + '=' * 30 + '\n' + message + '\n' + '=' * 30)
    return


def progress_bar(iteration, total_iterations):
    """
    Print a progress bar to the console
    :param iteration: current iteration
    :param total_iterations: total number of iterations
    :return: None
    """
    bar_length = 30

    percent = iteration / total_iterations
    percent_complete = int(percent * 100)

    progress = int(percent * bar_length)
    progress = '[' + '#' * progress + ' ' * (bar_length - progress) + ']'

    print(f'\r{progress} {percent_complete}% complete', end='', flush=True)
    return


def __calc_sparse_shape(array: np.ndarray, chunksize: int, verbose: bool) -> int:
    """
    Calculate the shape of the (pending) sparse array
    :param array: dense numpy array
    :param chunksize: chunksize to use for calculation - if None, will use the whole array
    :param verbose: whether to print progress statements
    :return: tuple of shape
    """

    data_shape = 0
    shape = array.shape

    if (chunksize is None) or (type(array) is np.ndarray):
        data_shape = np.count_nonzero(array)

    else:
        for i in range(0, shape[0], chunksize):
            progress_bar(i, shape[0]) if verbose else None
            data_shape += np.count_nonzero(array[i:i + chunksize])

    return data_shape


@njit
def __convert_to_sparse_data(sparse_coords, sparse_values, iteration: int, chunksize, sparse_coords_idx_baseline) -> (np.ndarray, np.ndarray):
    """
    Convert a chunk of a dense array to sparse data
    :param sparse_coords: sparse coordinates
    :param sparse_values: sparse values
    :param iteration: iteration number
    :param prev_sparse_coords_max: maximum idx of the previous sparse coordinates
    :return: tuple of sparse coordinates and sparse values
    """
    sparse_coords = list(sparse_coords)
    sparse_coords[0] += iteration
    sparse_coords_arr = np.empty((len(sparse_coords), sparse_coords[0].shape[0]), dtype=np.int64)
    for row in range(len(sparse_coords)):
        sparse_coords_arr[row] = sparse_coords[row]
    sparse_coords = sparse_coords_arr.T
    sparse_coords_dict = __create_sparse_coords_dictionary(sparse_coords, iteration, chunksize,
                                                           sparse_coords_idx_baseline)

    return sparse_coords, sparse_values, sparse_coords_dict


@njit(parallel=True)
def __create_sparse_coords_dictionary(sparse_coords, iteration, chunksize, sparse_coords_idx_baseline):
    """
    Convert a chunk of a dense array to sparse data
    :param sparse_coords: sparse coordinates
    :return: dictionary mapping dense rows to sparse coordinates
    """
    min_value = sparse_coords[:, 0].min()
    max_value = sparse_coords[:, 0].max()
    sparse_coords_dict = {i: (-1, -1) for i in range(iteration, iteration + chunksize)}

    for dense_row in prange(min_value, max_value + 1):
        sparse_to_dense_coords = np.where(sparse_coords[:, 0] == dense_row)[0]
        if np.any(sparse_to_dense_coords):
            sparse_coords_dict[dense_row] = (sparse_to_dense_coords.min() + sparse_coords_idx_baseline,
                                             sparse_to_dense_coords.max() + sparse_coords_idx_baseline)

    return sparse_coords_dict


def __write_sparse_arrays(array: np.ndarray or np.memmap, path: 'str', chunksize: int, verbose: bool) -> None:
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
    announce_progress('Identifying sparse shape...') if verbose else None
    nonzero_shape = __calc_sparse_shape(array, chunksize, verbose)

    # Create the sparse array binaries (memory-mapped)
    memmap_sparse_data = open_memmap(os.path.join(path, 'sparse_data.npy'),
                                     dtype=dense_dtype,
                                     mode='w+',
                                     shape=(nonzero_shape,))

    memmap_sparse_coords = open_memmap(os.path.join(path, 'sparse_coords.npy'),
                                       dtype=np.int32,
                                       mode='w+',
                                       shape=(nonzero_shape, len(dense_shape)))

    np.save(os.path.join(path, 'dense_shape.npy'), dense_shape)

    # Convert the dense array to sparse arrays
    announce_progress('Writing sparse arrays...') if verbose else None
    sparse_index = 0

    if (chunksize is None) or (type(array) is np.ndarray):
        # Use of the bool dtype accelerates this step
        sparse_coords = array.astype(bool).nonzero()
        sparse_values = array[sparse_coords]
        sparse_coords, sparse_values, sparse_coords_dict = __convert_to_sparse_data(sparse_coords, sparse_values, 0, 0)

        memmap_sparse_coords[:] = sparse_coords
        memmap_sparse_data[:] = sparse_values

    else:
        sparse_coords_dict = {}
        sparse_coords_idx_baseline = 0
        for chunk_idx in range(0, dense_shape[0], chunksize):
            progress_bar(chunk_idx, dense_shape[0]) if verbose else None
            array_chunk = array[chunk_idx:chunk_idx + chunksize]
            # Use of the bool dtype accelerates this step
            sparse_coords = array_chunk.astype(bool).nonzero()
            sparse_values = array_chunk[sparse_coords]
            sparse_coords, sparse_values, sparse_coords_dict_temp = __convert_to_sparse_data(sparse_coords, sparse_values,
                                                                                             chunk_idx, chunksize,
                                                                                             sparse_coords_idx_baseline)

            memmap_sparse_coords[sparse_index:sparse_index + sparse_coords.shape[0]] = sparse_coords
            memmap_sparse_data[sparse_index:sparse_index + sparse_coords.shape[0]] = sparse_values
            sparse_coords_dict.update(sparse_coords_dict_temp)

            sparse_index += sparse_coords.shape[0]
            sparse_coords_idx_baseline = sparse_coords_dict[min(chunk_idx + chunksize - 1, dense_shape[0]-1)][1] + 1

    with open(os.path.join(path, 'sparse_coords_dict.pkl'), 'wb') as f:
        pickle.dump(sparse_coords_dict, f)

    return


def to_sparse(array: np.ndarray or np.memmap, savepath: 'str', chunksize=1000, verbose=True) -> None:
    """
    Convert and write a dense array to a sparse array
    :param array: numpy array to be converted
    :param savepath: filepath to write sparse array to
    :param chunksize: number of memmap rows to process at a time if array is np.memmap - if None, will convert the whole array in memory
    :param verbose: whether to print progress statements
    :return: None
    """
    if not os.path.isdir(savepath):
        os.makedirs(savepath)

    __write_sparse_arrays(array, savepath, chunksize, verbose)
    return
