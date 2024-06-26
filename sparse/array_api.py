import numpy as np
from numba import njit, types, prange, typed
import pickle
import os
from numpy import ndarray


def load_sparse(data_path: str, coords_path=None, row_path=None, shape=None, sparse_reference_value=0):
    """
    Load a (memory-mapped) sparse array from disk
    :param data_path: path to sparse data array OR parent directory containing 'sparse_data.npy', 'sparse_coords.npy', and 'dense_shape.npy' arrays
    :param coords_path: (optional) path to sparse coordinates array
    :param row_path: (optional) path to dictionary mapping sparse to dense row coordinates
    :param shape: (optional) shape of the dense array, as either tuple or path to numpy array containing shape
    :param sparse_reference_value: (optional) value to be treated as sparse (default: 0)
    :return: SparseArray object
    """
    if os.path.isdir(data_path):
        coords_path = os.path.join(data_path, 'sparse_coords.npy')
        row_path = os.path.join(data_path, 'sparse_rows.npy')
        shape = os.path.join(data_path, 'dense_shape.npy')

        data_path = os.path.join(data_path, 'sparse_data.npy')

    return SparseArray(data_path, row_path, coords_path, shape, sparse_reference_value, mode='r')


class SparseArray:
    def __init__(self, data_path: str, row_path: str, coords_path, shape: str or tuple, sparse_reference_value, mode='r'):
        self.data = np.load(data_path, mmap_mode=mode)
        self.data_dtype = self.data.dtype

        self.rows = np.load(row_path, mmap_mode=mode)
        self.coords = np.load(coords_path, mmap_mode=mode)
        self.sparse_ref_value = sparse_reference_value

        if shape is tuple:
            self.shape, self.dense_shape = shape, shape
        else:
            self.shape, self.dense_shape = tuple(np.load(shape)), tuple(np.load(shape))

    def __getitem__(self, items):
        """"
        Get the values of the dense array at the given indices
        """
        if isinstance(items, (int, np.int_, np.uintc)):
            return self.__get_row(int(items))

        elif isinstance(items, slice):
            items = np.arange(*items.indices(self.dense_shape[0]))
            return self.__get_rows(items)

        elif isinstance(items, np.ndarray):
            if items.dtype == bool:
                if len(items) != self.dense_shape[0]:
                    raise ValueError(
                        'Boolean index shape mismatch: {} vs {}'.format(len(items), self.dense_shape[0]))
                else:
                    items = np.where(items)[0]
                    return self.__get_rows(items)
            elif items.dtype == int:
                if (items.ndim == 1) or (items.ndim == 2):
                    return self.__get_rows(items)
                else:
                    raise ValueError('Too many dimensions for boolean indexing: {}'.format(items.ndim))  # <- is this in the wrong place?????

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
            for item in items:
                if not isinstance(item, (int, slice, np.ndarray, type(Ellipsis), type(None))):
                    raise TypeError(
                        'indices must be integers, slices, np.ndarray, Ellipsis, or NoneType - not {}'.format(
                            type(item)))

            new_items = list(items)
            if items[0] is None:
                del new_items[0]
                return np.expand_dims(self.__getitem__(items[0])[tuple(items)], 0)

            elif isinstance(items[0], int):
                del new_items[0]

            if isinstance(items[0], int):
                del new_items[0]
            else:
                new_items[0] = slice(None, None, None)

            return self.__getitem__(items[0])[tuple(new_items)]

        else:
            raise TypeError(
                'indices must be integers, slices, np.ndarray, Ellipsis, or NoneType - not {}'.format(type(items)))
    """
    def __get_row(self, row_idx) -> np.ndarray:
        '''
        Get the values of the dense array at the given row indices
        :param row_idx: target row indices
        :return: dense numpy array values at the given row indices
        '''

        # Specify the search function based on the type of index given
        search_function = {int: self.__find_rows_int,
                           np.ndarray: self.__find_rows_ndarray}

        # Find the target coordinates for the given row indices
        find_coords, coords_present = search_function[type(row_idx)](self.coords_dict, row_idx)

        find_coords = np.concatenate(find_coords)
        find_coords = find_coords[find_coords >= 0]
        find_coords = np.sort(find_coords)

        # Assuming the row indices have non-zero data...
        if coords_present:
            # Locate the coordinates of the non-zero data in the original dense array
            target_coords = self.coords[find_coords]

            # Some slightly complex indexing to identify the non-zero cells of the (indexed) target array
            nonzero_idxs_in_row_idx = np.where(np.isin(row_idx, np.unique(target_coords[:, 0])))[0]
            location_of_each_new_idx_in_target_coords = np.unique(target_coords[:, 0], return_index=True)[1]

            target_coords[:, 0] = 0
            target_coords[location_of_each_new_idx_in_target_coords, 0] = nonzero_idxs_in_row_idx

            target_coords[:, 0] = np.maximum.accumulate(target_coords[:, 0], axis=0)

            # Re-create the dense array (of the shape requested) with all zeros
            target_data = np.zeros((1,) + self.dense_shape[1:], dtype=self.data_dtype) if isinstance(row_idx, int) else np.zeros(
                (len(row_idx),) + self.dense_shape[1:], dtype=self.data_dtype)

            # Fill the dense array with the non-zero data at the target coordinates
            target_data[tuple(target_coords.T)] = self.data[find_coords]

            # If only one row index was given, reduce the dimensionality
            if isinstance(row_idx, int):
                target_data = target_data[0]
        else:
            # If the row indices have no non-zero data, return an array of zeros
            target_data = np.zeros(self.dense_shape[1:], dtype=self.data_dtype) if isinstance(row_idx, int) else np.zeros(
                (len(row_idx),) + self.dense_shape[1:], dtype=self.data_dtype)

        return target_data
    """
    def __get_row(self, row_idx) -> np.ndarray:
        """
        Get the values of the dense array at the given row indices
        :param row_idx: target row indices
        :return: dense numpy array values at the given row indices
        """

        """
        CHECKPOINT 
        Need to find a quicker way of implementing the following code:

        np.concatenate([np.arange(i,j) for i,j in zip(self.rows[row_idx][:-1], self.rows[row_idx][1:])])

        Remember - self.rows[row_idx] gives the start and end indices of the coordinates for the self.data and self.coords arrays
        self.coords then needs to be converted using np.unravel_index(self.coords, self.dense_shape[1:])
        """
        # Finding a single row:
        start, end = self.rows[np.array([row_idx, row_idx+1])]
        data = self.data[start:end]
        flat_coords = self.coords[start:end]
        coords = np.unravel_index(flat_coords, self.dense_shape[1:])

        indexed_data = np.zeros(self.dense_shape[1:], dtype=self.data_dtype)
        indexed_data[coords] = data

        return indexed_data

    def __get_rows(self, row_idxs) -> np.ndarray:
        """
        Get the values of the dense array at the given row indices
        :param row_idx: target row indices
        :return: dense numpy array values at the given row indices
        """

        """
        CHECKPOINT 
        Need to find a quicker way of implementing the following code:

        np.concatenate([np.arange(i,j) for i,j in zip(self.rows[row_idx][:-1], self.rows[row_idx][1:])])

        Remember - self.rows[row_idx] gives the start and end indices of the coordinates for the self.data and self.coords arrays
        self.coords then needs to be converted using np.unravel_index(self.coords, self.dense_shape[1:])
        """
        # Finding multiple rows:

        # Generate the coordinate (reference) index pointers
        lengths, coord_idxs = zip(*[(end - start, np.arange(start, end, dtype=np.uint64))
                                    for start, end in self.rows[np.hstack((row_idxs.reshape(-1, 1),
                                                                           row_idxs.reshape(-1, 1)+1))]])
        coord_idxs = np.concatenate(coord_idxs)

        # Retrieve the data
        data = self.data[coord_idxs]
        flat_coords = self.coords[coord_idxs]
        coords = np.unravel_index(flat_coords, self.dense_shape[1:])

        # Generate the blank (sparse) matrix to be filled
        indexed_data = np.ones((len(lengths),) + self.dense_shape[1:], dtype=self.data_dtype) * self.sparse_ref_value

        # Generate the dense indices for the data
        coords = (np.repeat(np.arange(len(lengths)), np.array(lengths).astype(np.int64)),) + coords

        indexed_data[coords] = data

        return indexed_data

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

    def __len__(self):
        return self.shape[0]

