import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm
import os
from numba import njit, types, typed, prange
import pickle
from math import ceil

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


@njit(parallel=True)
def __calc_sparse_shape(array: np.ndarray, chunksize: int) -> tuple:
    """
    Calculate the shape of the (pending) sparse array
    :param array: dense numpy array
    :param chunksize: chunksize to use for calculation - if None, will use the whole array
    :param verbose: whether to print progress statements
    :return: tuple of shape
    """
    data_shape = 0
    shape = array.shape

    for i in prange(shape[0]):
        data_shape += np.count_nonzero(array[i])

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
    sparse_coords_dict = {}

    for dense_row in range(sparse_coords[:, 0].min(), sparse_coords[:, 0].max() + 1):
        sparse_to_dense_coords = np.where(sparse_coords[:, 0] == dense_row)[0]
        if any(sparse_to_dense_coords):
            sparse_coords_dict[dense_row] = (np.where(sparse_coords[:, 0] == dense_row)[0].min(),
                                             np.where(sparse_coords[:, 0] == dense_row)[0].max())
        else:
            sparse_coords_dict[dense_row] = (-1, -1)

    return sparse_coords, sparse_values, sparse_coords_dict


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
    nonzero_shape = __calc_sparse_shape(array)

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

    if (chunksize is None) or (type(array) is np.ndarray):
        sparse_coords, sparse_values, sparse_coords_dict = __convert_to_sparse_data(array, 0)

        memmap_sparse_coords[:] = sparse_coords
        memmap_sparse_data[:] = sparse_values
        with open(os.path.join(path, 'sparse_coords_dict.pkl'), 'wb') as f:
            pickle.dump(sparse_coords_dict, f)

    else:
        sparse_coords_dict = {}
        for chunk_idx in tqdm(range(0, dense_shape[0], chunksize)) if verbose else range(0, dense_shape[0], chunksize):
            sparse_coords, sparse_values, sparse_coords_dict_iter = __convert_to_sparse_data(
                array[chunk_idx:chunk_idx + chunksize], chunk_idx)

            memmap_sparse_coords[sparse_index:sparse_index + sparse_coords.shape[0]] = sparse_coords
            memmap_sparse_data[sparse_index:sparse_index + sparse_coords.shape[0]] = sparse_values

            sparse_index += sparse_coords.shape[0]
            sparse_coords_dict.update(sparse_coords_dict_iter)

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
