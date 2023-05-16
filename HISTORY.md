#Changelog

All notable changes to the `PySparse` package will be documented in this file.

## [Unreleased]

## [0.1.4] - 2023-05-16

### Added
- N/A

### Fixed
- N/A

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
