from sparse import load_sparse, to_sparse
import numpy as np

temp = load_sparse('/Users/thomasfrost/Oxford:UCL Work/Year 11 PhD 1 - 2022:2023/PhD Research/physionet.org/MIMIC_Processing/data/icu/processing/train/insulin_rate_arrays (sparse)/train/data/deleteme')

idx = np.unique(np.random.default_rng().integers(0, 1000, size=150))[:100]
print(temp[idx])

