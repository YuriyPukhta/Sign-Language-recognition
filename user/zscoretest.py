from scipy.stats import zscore
import numpy as np

ns = np.array([2, 1, 1, 2, 3])
ns2 = ns / np.sum(ns)
print(zscore(ns2))