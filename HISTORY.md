#Changelog

All notable changes to the `PySparse` package will be documented in this file.

## [Unreleased]

## [0.1.5] - 2023-05-18

### Added
- N/A

### Fixed
- N/A

### Changed
- Further optimised indexing through use of parallel find_indices search, with x2 speed-up.

### Removed
- N/A

## [0.1.4] - 2023-05-16

### Added
- load_sparse() can now take either a single path to the parent directory of the three arrays, or individual paths to each
- if using a single path to parent directory to load the sparse arrays, the files must be kept named as the standard 'sparse_data.npy', 'sparse_coords.npy', and 'dense_shape.npy' filenames

### Fixed
- Fixed issues with indexing for 1D arrays / rows where the values are all zero

### Changed
- to_sparse() default behaviour will now convert an array in one go if the array is already all in memory (i.e., np.ndarray), or else via chunksize if the array is memory-mapped (np.memmap).

### Removed
- N/A


## [0.1.3] - 2023-05-15

### Added
- N/A

### Fixed
- Reduced dependency for numba to <0.57 due to numba/numpy errors.
- Fixed missing parameter in __calc_sparse_shape()

### Changed
- Changed __init__.py for a cleaner interface - just two options! (to_sparse and load_sparse)
- to_sparse() now creates directory as part of savepath

### Removed
- N/A

## [0.1.2] - 2023-05-15

### Added
- N/A

### Fixed
- N/A

### Changed
- Re-uploaded due to broken compile

### Removed
- N/A

## [0.1.1] - 2023-05-15

### Added
- N/A

### Fixed
- Fixed issues with dependences

### Changed
- N/A

### Removed
- N/A

## [0.1.0] - 2023-05-15

### Added
- Initial release of `PySparse`

### Fixed
- N/A

### Changed
- N/A

### Removed
- N/A
