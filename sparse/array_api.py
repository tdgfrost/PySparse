import numpy as np
from numba import njit, types, prange, typed
import pickle
import os
from numpy import ndarray


def load_sparse(data_path: str, coords_path=None, coords_dict_path = None, shape=None):
    """
    Load a (memory-mapped) sparse array from disk
    :param data_path: path to sparse data array OR parent directory containing 'sparse_data.npy', 'sparse_coords.npy', and 'dense_shape.npy' arrays
    :param coords_path: (optional) path to sparse coordinates array
    :param coords_dict_path: (optional) path to dictionary mapping sparse to dense row coordinates
    :param shape: (optional) shape of the dense array, as either tuple or path to numpy array containing shape
    :return: SparseArray object
    """
    if os.path.isdir(data_path):
        coords_path = os.path.join(data_path, 'sparse_coords.npy')
        coords_dict_path = os.path.join(data_path, 'sparse_coords_dict.pkl')
        shape = os.path.join(data_path, 'dense_shape.npy')
        data_path = os.path.join(data_path, 'sparse_data.npy')

    return SparseArray(data_path, coords_path, coords_dict_path, shape, mode='r')


class SparseArray:
    def __init__(self, data_path: str, coords_path: str, coords_dict_path, shape: str or tuple, mode='r'):
        self.data = np.load(data_path, mmap_mode=mode)
        self.data_dtype = self.data.dtype

        self.coords = np.load(coords_path, mmap_mode=mode)
        self.coords_shape = self.coords.shape
        with open(coords_dict_path, 'rb') as f:
            self.coords_dict = typed.Dict.empty(key_type=types.int64, value_type=types.UniTuple(types.int64, count=2))
            coords_dict = pickle.load(f)
            for key, value in coords_dict.items():
                self.coords_dict[key] = value

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
        find_coords, coords_present = search_function[type(row_idx)](self.coords_dict, row_idx)

        find_coords = np.concatenate(find_coords)
        find_coords = find_coords[find_coords >= 0]

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
    @njit(types.Tuple((types.List(types.Array(types.int64, 1, 'C')),
                       types.boolean))(types.DictType(types.int64, types.UniTuple(types.int64, count=2)),
                                       types.int64))
    def __find_rows_int(coords_dict, row_idx: int) -> tuple[list[ndarray], bool]:
        """
        Efficiently search the coordinates to find the indices that match the given row indices
        :param coords_dict: dictionary mapping each dense row index to the first/last indices of the coordinates
        :param row_idx: target row index (integer)
        :return: numpy array of coordinates that match the given row indices
        """

        start, end = coords_dict[row_idx]
        find_coords = [np.arange(start, end+1)] if start >= 0 else [np.array([-1])]
        coords_present = True if start >= 0 else False

        return find_coords, coords_present

    """
        @njit(types.Tuple((types.List(types.Array(types.int64, 1, 'C')), types.boolean))(types.DictType(types.int64, types.UniTuple(types.int64, count=2)),
                                                                                         types.Array(types.int64, 1, 'A')),
              parallel=True)
    """
    @staticmethod
    def __find_rows_ndarray(coords_dict, row_idx: np.ndarray) -> tuple[list[ndarray], bool]:
        """
        Efficiently search the coordinates to find the indices that match the given row indices
        :param coords_dict: dictionary mapping each dense row index to the first/last indices of the coordinates
        :param row_idx: target row indices (numpy array)
        :return: numpy array of coordinates that match the given row indices
        """

        find_coords = [np.array([-1]) for _ in range(len(row_idx))]
        coords_present = False

        for row in prange(len(row_idx)):
            start, end = coords_dict[row_idx[row]]
            find_coords[row] = np.arange(start, end+1) if start >= 0 else np.array([-1])

            coords_present += True if start >= 0 else False

        coords_present = bool(coords_present)

        return find_coords, coords_present

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

