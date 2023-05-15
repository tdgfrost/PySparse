import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm
import os
from numba import njit, types


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
    nonzero_shape = __calc_sparse_shape(array, verbose)

    # Create directory for the destination arrays
    sparse_path = os.path.join(path, 'sparse_arrays')
    os.mkdir(sparse_path)

    # Create the sparse array binaries (memory-mapped)
    memmap_sparse_data = open_memmap(os.path.join(sparse_path, 'sparse_data.npy'),
                                     dtype=dense_dtype,
                                     mode='w+',
                                     shape=nonzero_shape)

    memmap_sparse_coords = open_memmap(os.path.join(sparse_path, 'sparse_coords.npy'),
                                       dtype=np.int32,
                                       mode='w+',
                                       shape=(nonzero_shape[0], len(dense_shape)))

    np.save(os.path.join(sparse_path, 'dense_shape.npy'), dense_shape)

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
    __write_sparse_arrays(array, savepath, chunksize, verbose)
    return


"""
============================================================
SparseArray class and functions
============================================================
"""

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


class SparseArray:
    def __init__(self, directory_path, mode='r'):
        self.dense_shape = tuple(np.load(os.path.join(directory_path, 'target_dense_shape.npy')))
        self.coords = np.load(os.path.join(directory_path, 'sparse_coords.npy'), mmap_mode=mode)
        self.data = np.load(os.path.join(directory_path, 'sparse_data.npy'), mmap_mode=mode)

    def __getitem__(self, items):
        """"
        Get the values of the dense array at the given indices
        """
        if isinstance(items, int):
            return self.__get_row(items)

        elif isinstance(items, slice):
            items = np.arange(*items.indices(self.dense_shape[0]))
            return self.__get_row(items)

        elif isinstance(items, np.ndarray):
            if items.dtype == bool:
                if len(items) != self.dense_shape[0]:
                    raise ValueError(
                        'Boolean index shape mismatch: {} vs {}'.format(len(items), self.dense_shape[0]))
                else:
                    items = np.where(items)[0]
                    return self.__get_row(items)
            elif items.dtype == int:
                if (items.ndim == 1) or (items.ndim == 2):
                    return self.__get_row(items)
                else:
                    raise ValueError('Too many dimensions for boolean indexing: {}'.format(items.ndim))

            else:
                raise TypeError('arrays used as indices must be of integer (or boolean) type')

        elif items is Ellipsis:
            target_data = np.zeros((1,) + self.dense_shape[1:])
            target_data[tuple(self.coords.T)] = self.data
            return target_data

        elif items is None:
            target_data = np.zeros((1,) + self.dense_shape[1:])
            target_data[tuple(self.coords.T)] = self.data
            shape = (1,) + self.dense_shape
            return target_data.reshape(shape)

        elif isinstance(items, tuple):
            items = list(items)
            for idx, item in enumerate(items):
                if not isinstance(item, (int, slice, np.ndarray, type(Ellipsis), type(None))):
                    raise TypeError(
                        'indices must be integers, slices, np.ndarray, Ellipsis, or NoneType - not {}'.format(
                            type(item)))

            if isinstance(items[0], slice):
                items[0] = np.arange(*items[0].indices(self.dense_shape[0]))
                return self.__get_item(items)

            elif isinstance(items[0], np.ndarray):
                if items[0].dtype == bool:
                    if len(items[0]) != self.dense_shape[0]:
                        raise ValueError(
                            'Boolean index shape mismatch: {} vs {}'.format(len(items[0]), self.dense_shape[0]))
                    else:
                        items[0] = np.where(items[0])[0]
                        return self.__get_item(items)

            elif (items[0] is Ellipsis) or (items[0] is None):
                target_data = np.zeros((1,) + self.dense_shape[1:])
                target_data[tuple(self.coords.T)] = self.data
                return target_data[tuple(items)]

            return self.__get_item(items)

        else:
            raise TypeError(
                'indices must be integers, slices, np.ndarray, Ellipsis, or NoneType - not {}'.format(type(items)))

    def __get_row(self, row_idx) -> np.ndarray:
        """
        Get the values of the dense array at the given row indices
        :param row_idx: target row indices
        :return: dense numpy array values at the given row indices
        """

        # Specify the search function based on the type of index given
        search_function = {int: self.__find_rows_int,
                           np.ndarray: self.__find_rows_ndarray}

        # Find the target coordinates for the given row indices
        find_coords, coords_present = search_function[type(row_idx)](self.coords, row_idx,
                                                                     maxlen=self.coords_shape[0])

        # Assuming the row indices have non-zero data...
        if coords_present:
            # Locate the coordinates of the non-zero data in the original dense array
            target_coords = self.coords[find_coords]
            target_coords[:, 0] -= target_coords[:, 0].min()

            # Re-create the dense array (of the shape requested) with all zeros
            target_data = np.zeros((1,) + self.dense_shape[1:]) if isinstance(row_idx, int) else np.zeros(
                (len(row_idx),) + self.dense_shape[1:])

            # Fill the dense array with the non-zero data at the target coordinates
            target_data[tuple(target_coords.T)] = self.data[find_coords]
        else:
            # If the row indices have no non-zero data, return an array of zeros
            target_data = np.zeros((1,) + self.dense_shape[1:]) if isinstance(row_idx, int) else np.zeros(
                (len(row_idx),) + self.dense_shape[1:])

        return target_data

    @staticmethod
    @njit(types.Tuple((types.int64[:], types.boolean))(types.Array(types.int32, 2, 'C', readonly=True), types.int64,
                                                       types.int64))
    def __find_rows_int(coords: np.ndarray, row_idx: int, maxlen: int) -> types.Tuple:
        """
        Efficiently search the coordinates to find the indices that match the given row index
        :param coords: numpy 2D array of coordinates
        :param row_idx: target row index (integer)
        :param maxlen: maximum length of the coordinates array
        :return: numpy array of coordinates that match the given row index
        """

        # Start with an efficient binary tree search algorithm for identifying the start and end indices
        start_point, end_point = find_indices(coords, row_idx, 0, maxlen - 1)

        coords_present = True if start_point >= 0 else False

        return np.array([i for i in range(start_point, end_point + 1)]), coords_present

    @staticmethod
    @njit(types.Tuple((types.int64[:], types.boolean))(types.Array(types.int32, 2, 'C', readonly=True),
                                                       types.Array(types.int64, 1, 'A'), types.int64))
    def __find_rows_ndarray(coords: np.ndarray, row_idx: np.ndarray, maxlen: int) -> (np.ndarray, bool):
        """
        Efficiently search the coordinates to find the indices that match the given row indices
        :param coords: numpy 2D array of coordinates
        :param row_idx: target row indices (numpy array)
        :param maxlen: maximum length of the coordinates array
        :return: numpy array of coordinates that match the given row indices
        """

        find_coords = []

        for row in np.sort(row_idx):
            # Start with an efficient binary tree search algorithm for identifying the start and end indices
            start_point, end_point = find_indices(coords, row, 0, maxlen - 1)
            find_coords.extend([i for i in range(start_point, end_point + 1)]) if start_point >= 0 else None

        coords_present = True if find_coords else False

        return np.array(find_coords), coords_present

    def __get_item(self, items) -> np.ndarray:
        """
        Get the values of the dense array at the given indices
        :param items: tuple of indices
        :return: dense numpy array values at the given indices
        """

        # Start by getting the values of the dense array at the given row indices,
        # with the dense array fully in memory
        target_data = self.__get_row(items[0])

        # Then index the dense array with the remaining indices as usual
        items[0] = slice(None)
        target_data = target_data[tuple(items)]

        return target_data

