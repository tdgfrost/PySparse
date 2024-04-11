import numpy as np
from numpy.lib.format import open_memmap
import os
from numba import prange

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


def find_smallest_dtype(max_value: int) -> np.dtype:
    """
    Find the smallest numpy dtype that can hold the given value
    :param max_value: maximum value to be held by the dtype
    :return: numpy dtype
    """
    if max_value <= np.iinfo(np.uint8).max:
        return np.uint8
    elif max_value <= np.iinfo(np.uint16).max:
        return np.uint16
    elif max_value <= np.iinfo(np.uint32).max:
        return np.uint32
    else:
        return np.uint64


def __calc_sparse_shape(array: np.ndarray, chunksize: int, sparse_ref_value, verbose: bool) -> int:
    """
    Calculate the shape of the (pending) sparse array
    :param array: dense numpy array
    :param chunksize: chunksize to use for calculation - if None, will use the whole array
    :param sparse_ref_value: value to be considered as sparse (default is 0)
    :param verbose: whether to print progress statements
    :return: tuple of shape
    """

    data_shape = 0
    shape = array.shape

    announce_progress('Counting non-sparse values...')
    if chunksize is None:
        data_shape = np.sum(array[:] != sparse_ref_value) if not np.isnan(sparse_ref_value) else np.sum(~np.isnan(array))

    else:
        for i in range(0, shape[0], chunksize):
            progress_bar(i, shape[0] // chunksize * chunksize) if verbose else None
            data_shape += np.sum(array[i:i + chunksize] != sparse_ref_value) if not np.isnan(sparse_ref_value) \
                else np.sum(~np.isnan(array[i:i + chunksize]))

    return data_shape


def __write_sparse_arrays(array: np.ndarray or np.memmap, path: 'str', chunksize: int,
                          sparse_ref_value, verbose: bool) -> None:
    """
    Simultaneously convert and write a dense array to sparse arrays
    :param array: dense numpy array to be converted
    :param path: path to write sparse arrays to
    :param chunksize: chunksize to use for conversion - if None, will convert the whole array in memory
    :param sparse_ref_value: value to be considered as sparse (default is 0)
    :param verbose: whether to print progress statements
    :return:
    """

    # Identify the relevant shapes of the dense and sparse arrays
    dense_shape = array.shape
    dense_dtype = array.dtype
    announce_progress('Identifying sparse shape...') if verbose else None
    if chunksize > dense_shape[0]:
        chunksize = dense_shape[0]
    nonsparse_shape = __calc_sparse_shape(array, chunksize, sparse_ref_value, verbose)

    """
    Okay, idea:
    Could stick with CSR, but to adapt matrices more than 2D, we simply take all the columns+ and flatten
    them using np ravel and unravel multi index.
    So a 3D array of shape (a x b x c) becomes (a x d), where d is a flattened version of b and c.
    This remains optimised if the row is the primary unit of searching, which it will be here!
    
    Note for later (may or may not be helpful).
    Let's say you've compressed the rows AND the columns.
    If you search for row 1000, that's easy to do.
    What about decoding subequent dimensions?
    Steps, using the example `query = [:, :, 10:]`, of shape `dense_shape = (S x M x N)`
    1) Generate an index
    > index = np.indices(dense_shape)
    
    2) Convert this to a flattened query using ravel_multi_index:
    > flattened_index = np.ravel_multi_index(index[query], dims=dense_shape)
    
    3) Use this to find the appropriate items from the compressed column
    """
    # Create the sparse array binaries (memory-mapped)
    memmap_sparse_data = open_memmap(os.path.join(path, 'sparse_data.npy'),
                                     dtype=dense_dtype,
                                     mode='w+',
                                     shape=(nonsparse_shape,))

    memmap_sparse_rows = open_memmap(os.path.join(path, 'sparse_rows.npy'),
                                     dtype=find_smallest_dtype(nonsparse_shape),
                                     mode='w+',
                                     shape=(dense_shape[0],))

    memmap_sparse_coords = open_memmap(os.path.join(path, 'sparse_coords.npy'),
                                       dtype=find_smallest_dtype(np.prod(dense_shape[1:])),
                                       mode='w+',
                                       shape=(nonsparse_shape,))

    np.save(os.path.join(path, 'dense_shape.npy'), dense_shape)

    # Convert the dense array to sparse arrays
    announce_progress('Writing sparse arrays...') if verbose else None

    if chunksize is None:
        sparse_values_idx = np.where(array[:] != sparse_ref_value) if not np.isnan(sparse_ref_value) \
            else np.where(~np.isnan(array[:]))

        # code for the rows
        compressed_sparse_rows = np.cumsum(np.bincount(sparse_values_idx[0])[:-1])
        compressed_sparse_rows = np.insert(compressed_sparse_rows, 0, 0)
        memmap_sparse_rows[:] = compressed_sparse_rows

        # code for the cols
        sparse_coords_flattened = np.ravel_multi_index(sparse_values_idx[1:], dims=(dense_shape[1:]))
        memmap_sparse_coords[:] = sparse_coords_flattened

    else:
        sparse_save_idx = 0
        last_sparse_row_max = 0
        for dense_chunk_idx in range(0, dense_shape[0], chunksize):
            progress_bar(dense_chunk_idx, dense_shape[0] // chunksize * chunksize) if verbose else None
            next_dense_chunk_idx = dense_chunk_idx + chunksize

            # get array chunk
            array_chunk = array[dense_chunk_idx:next_dense_chunk_idx]

            # get index
            sparse_values_idx = np.where(array_chunk != sparse_ref_value) if not np.isnan(sparse_ref_value) \
                else np.where(~np.isnan(array_chunk))

            # get values
            sparse_values = array_chunk[sparse_values_idx]
            next_sparse_save_idx = sparse_save_idx + sparse_values.size
            memmap_sparse_data[sparse_save_idx:next_sparse_save_idx] = sparse_values

            # code for the rows
            sparse_rows = np.cumsum(np.bincount(sparse_values_idx[0]))
            sparse_rows = np.insert(sparse_rows, 0, 0) + last_sparse_row_max
            sparse_rows, last_sparse_row_max = sparse_rows[:-1], sparse_rows[-1]
            memmap_sparse_rows[dense_chunk_idx:next_dense_chunk_idx] = sparse_rows

            # code for the cols
            sparse_coords_flattened = np.ravel_multi_index(sparse_values_idx[1:], dims=(dense_shape[1:]))
            memmap_sparse_coords[sparse_save_idx:next_sparse_save_idx] = sparse_coords_flattened

            sparse_save_idx = next_sparse_save_idx

    return


def to_sparse(array: np.ndarray or np.memmap, savepath: 'str', chunksize=1000, sparse_reference_value=0, verbose=True) -> None:
    """
    Convert and write a dense array to a sparse array
    :param array: numpy array to be converted
    :param savepath: filepath to write sparse array to
    :param chunksize: number of memmap rows to process at a time if array is np.memmap - if None, will convert the whole array in memory
    :param sparse_reference_value: value to be considered as sparse (default is 0)
    :param verbose: whether to print progress statements
    :return: None
    """
    if not os.path.isdir(savepath):
        os.makedirs(savepath)

    if not isinstance(sparse_reference_value, int) and not isinstance(sparse_reference_value, float) and not np.isnan(sparse_reference_value):
        raise ValueError('Sparse value must be an integer, float, or NaN')

    __write_sparse_arrays(array, savepath, chunksize, sparse_reference_value, verbose)
    return
