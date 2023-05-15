This package exists for one purpose: to encode and decode sparse numpy arrays into a more space-efficient format. Its target audience is users requiring rapid access to massive arrays without memory issues.

Ordinarily, a numpy array might be stored in one of a couple of ways:
1) As a binary
2) As a compressed file format (e.g., h5df)

The advantage of binary is that the file can be memory mapped, allowing access to sections of the array without out-of-memory (OOM) issues. The disadvantage is that such binaries are, necessarily, of a very large file size.

The advantage of compressed file formats is that they provide a much smaller disk space footprint, but at the cost of slower file access. Even h5df file formats, which can provide memory-mapped services of their own, will still be necessarily slow because the non-contiguous array has to be searched to find the relevant indices.

PySparse takes a given numpy array, and encodes it into two (smaller) arrays - a 1D array containing the non-zero values, and a 2D array containing the mapped coordinates for these values in the original 'dense' numpy array.

The resultant arrays can then be re-loaded into the SparseArray class, from which indexing can be performed to return to the original dense array.

SparseArray is NOT a child class of Numpy, and numpy functions cannot be performed directly on it (for now). To do this, you would need to manually implement a solution yourself e.g., decoding the entire array at once using [:], or using a for-loop to iterate through the SparseArray in chunks to avoid OOM errors. 
