
import numpy as np
import pandas as pd

obj = pd.read_pickle("/content/HERA_04-03-2022_all.pkl")
assert isinstance(obj, list) and len(obj) == 4, f"unexpected obj type/len: {type(obj)}, {len(obj)}"

X0, X1, X2, X3 = obj

cands = [x for x in obj if isinstance(x, np.ndarray) and x.ndim == 4]
assert len(cands) >= 1, "No 4D ndarray found"
X = max(cands, key=lambda a: a.size)

sample_idx = 0
D_real = X[sample_idx, :, :, 0]
D_real = D_real.astype(np.float32)
