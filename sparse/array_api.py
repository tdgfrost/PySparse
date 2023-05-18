import numpy as np
from numba import njit, types, prange, typed
from numpy import ndarray

from .core import find_indices
import os


def load_sparse(data_path: str, coords_path=None, shape=None):
    """
    Load a (memory-mapped) sparse array from disk
    :param data_path: path to sparse data array OR parent directory containing 'sparse_data.npy', 'sparse_coords.npy', and 'dense_shape.npy' arrays
    :param coords_path: (optional) path to sparse coordinates array
    :param shape: (optional) shape of the dense array, as either tuple or path to numpy array containing shape
    :return: SparseArray object
    """
    if os.path.isdir(data_path):
        coords_path = os.path.join(data_path, 'sparse_coords.npy')
        shape = os.path.join(data_path, 'dense_shape.npy')
        data_path = os.path.join(data_path, 'sparse_data.npy')

    return SparseArray(data_path, coords_path, shape, mode='r')


class SparseArray:
    def __init__(self, data_path: str, coords_path: str, shape: str or tuple, mode='r'):
        self.data = np.load(data_path, mmap_mode=mode)
        self.data_dtype = self.data.dtype

        self.coords = np.load(coords_path, mmap_mode=mode)
        self.coords_shape = self.coords.shape

        if shape is tuple:
            self.dense_shape = shape
        else:
            self.dense_shape = tuple(np.load(shape))

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

        find_coords = np.concatenate([np.arange(start, end+1) for start, end in find_coords if start != -1])

        # Assuming the row indices have non-zero data...
        if coords_present:
            # Locate the coordinates of the non-zero data in the original dense array
            target_coords = self.coords[find_coords]

            # Some slightly complex indexing to identify the non-zero cells of the (indexed) target array
            nonzero_row_indices_unique = np.where(np.isin(row_idx, np.unique(target_coords[:, 0])))[0]
            nonzero_row_indices_unique_idx = np.unique(target_coords[:, 0], return_index=True)[1]

            target_coords[:, 0] = -1
            target_coords[nonzero_row_indices_unique_idx, 0] = nonzero_row_indices_unique

            target_coords[:, 0] = np.maximum.accumulate(target_coords[:, 0], axis=0)

            # Re-create the dense array (of the shape requested) with all zeros
            target_data = np.zeros((1,) + self.dense_shape[1:], dtype=self.data_dtype) if isinstance(row_idx, int) else np.zeros(
                (len(row_idx),) + self.dense_shape[1:], dtype=self.data_dtype)

            # Fill the dense array with the non-zero data at the target coordinates
            target_data[tuple(target_coords.T)] = self.data[find_coords]
        else:
            # If the row indices have no non-zero data, return an array of zeros
            target_data = np.zeros((1,) + self.dense_shape[1:], dtype=self.data_dtype) if isinstance(row_idx, int) else np.zeros(
                (len(row_idx),) + self.dense_shape[1:], dtype=self.data_dtype)

        return target_data

    @staticmethod
    @njit(types.Tuple((types.int64[:], types.boolean))(types.Array(types.int32, 2, 'C', readonly=True), types.int64,
                                                       types.int64))
    def __find_rows_int(coords: np.ndarray, row_idx: int, maxlen: int) -> tuple[ndarray, bool]:
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
    @njit(types.Tuple((types.ListType(types.UniTuple(types.int64, 2)), types.boolean))(types.Array(types.int32, 2, 'C', readonly=True),
                                                                                       types.Array(types.int64, 1, 'A'), types.int64),
          parallel=True)
    def __find_rows_ndarray(coords: np.ndarray, row_idx: np.ndarray, maxlen: int) -> (np.ndarray, bool):
        """
        Efficiently search the coordinates to find the indices that match the given row indices
        :param coords: numpy 2D array of coordinates
        :param row_idx: target row indices (numpy array)
        :param maxlen: maximum length of the coordinates array
        :return: numpy array of coordinates that match the given row indices
        """

        find_coords = typed.List([(-1, -1) for _ in range(len(row_idx))])

        for row in prange(len(row_idx)):
            # Start with an efficient binary tree search algorithm for identifying the start and end indices
            start_point, end_point = find_indices(coords, row_idx[row], 0, maxlen - 1)
            find_coords[row] = (start_point, end_point)

        return find_coords, True

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
        items[0] = Ellipsis
        target_data = target_data[tuple(items)]

        return target_data

