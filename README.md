# PySparse

This package exists for one purpose: to encode and decode sparse NumPy arrays into a more space-efficient format. Its target audience is users requiring rapid access to massive arrays without out-of-memory issues.

For machine-learning purposes, a NumPy array might be stored in one of two formats:
1) As a binary (extremely large, very fast to access)
2) As a compressed file format like HDF5 (much smaller, much slower to access)

Both of the above can be accessed in a memory-mapped fashion (to avoid OOM errors), but both require concessions in the form of either size or speed.

PySparse tries to find a middle ground. The package takes a given NumPy array, and encodes it into four files:
- a 1D array containing all non-zero values
- a 2D array containing the mapped coordinates for these values in the original 'dense' array
- a 1D array containing the shape of the original dense array
- and a pickled dictionary mapping dense-to-sparse row coordinates, new to v1.

These arrays can then be re-loaded into a SparseArray class, from which indexing can be performed as if you were operating on the original dense array. PySparse will decode the desired indices on-the-fly using memory-mapping of the encoded coordinates, and provide the desired subarray in-memory.

# Example Code

PySparse only provides you with two functions - one to encode an array, and one to load the encoded arrays.

**Encoding an Array**

```
> from sparse import to_sparse, load_sparse
> import os

> print(to_sparse.__doc__)

    Convert and write a dense array to a sparse array
    :param array: numpy array to be converted
    :param savepath: filepath to write sparse array to
    :param chunksize: number of memmap rows to process at a time if array is np.memmap - if None, will convert the whole array in memory
    :param verbose: whether to print progress statements
    :return: None

> array = np.random.default_rng().integers(low=0, high=5, size=(10000))
> to_sparse(array=array,
	        savepath='/.',
	        chunksize=None,
	        verbose=True)

==================================================
Identifying sparse shape...
==================================================
==================================================
Writing sparse arrays...
==================================================

> print(os.listdirs(savepath))

['dense_shape.npy', 'sparse_coords_dict.pkl', 'sparse_coords.npy', 'sparse_data.npy']

```
**Decoding an Array**
```
> print(load_sparse.__doc__)

    Load a (memory-mapped) sparse array from disk
    :param data_path: path to sparse data array OR parent directory containing 'sparse_data.npy', 'sparse_coords.npy', 'dense_shape.npy', and 'sparse_coords_dict.pkl' files
    :param coords_path: (optional) path to sparse coordinates array
    :param coords_dict_path: (optional) path to dictionary mapping sparse to dense row coordinates
    :param shape: (optional) shape of the dense array, as either tuple or path to numpy array containing shape
    :return: SparseArray object

> encoded_array = load_sparse(data_path='./')
> print(encoded_array[100:110])

array([3, 0, 4, 4, 2, 2, 3, 4, 0, 0])

> print(array[100:110])

array([3, 0, 4, 4, 2, 2, 3, 4, 0, 0])

> print(np.array_equal(array, encoded_array[:]))

True
```
# Benchmarks

Sample data is a sparse array of size (1102729, 288, 63).

**Relative Sizes**
- .npy binary = 80GB
- HDF5 with maximal gzip compression = 1.79GB
- SparseArray directory = 11.77GB

**Loading first 10,000 rows**
- .npy binary (memory-mapped) = 39.3 µs (757 ns with cache)
- HDF5 = 957 ms (898 ms with cache)
- SparseArray = 445 ms (402 ms with cache)

**Loading random 10,000 rows**
- .npy binary (memory-mapped) = 4.24 seconds (88 ms with cache)
- HDF5 = 32 seconds (31 seconds with cache)
- SparseArray = 2.28 seconds (320 ms with cache)

Of course, there are so many caveats to this (sparsity of data, shape of data, computational power available, etc), so you'll just have to try it yourself to see whether it works for you.

## Important Considerations
- SparseArray is NOT a child class of NumPy, and you can't perform functions directly on it. All operations must be done following an index call.
- SparseArray is NOT a stable package. It is cobbled together using a mix of other packages, and you should always keep a copy of your data in a gold-standard format (e.g., HDF5). You should also perform basic sanity-checks of your own to ensure the encoded array does match the original as expected.
- At present, SparseArray is read-only and does NOT support in-place modifying of the data. If you wish to change the data, you will need to change the gold-standard and re-encode your data from scratch.
