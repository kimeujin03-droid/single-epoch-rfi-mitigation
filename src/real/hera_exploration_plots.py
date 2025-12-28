
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# For standalone execution, load D_real here
obj = pd.read_pickle("/content/HERA_04-03-2022_all.pkl")
X = max([x for x in obj if isinstance(x, np.ndarray) and x.ndim == 4], key=lambda a: a.size)
sample_idx = 0
D_real = np.squeeze(X[sample_idx, :, :, 0]).astype(np.float32)

A = D_real.astype(np.float32)

# (1) 행/열 평균 스펙트럼 비교
row_mean = np.mean(A, axis=0)
col_mean = np.mean(A, axis=1)

plt.figure(figsize=(10,3))
plt.plot(row_mean, label="mean over axis=0")
plt.plot(col_mean, label="mean over axis=1")
plt.legend(); plt.title("Axis-mean comparison"); plt.tight_layout()
plt.savefig("axis_mean_compare.png", dpi=180); plt.close()

# (2) 대각선 구조 확인 (freq-freq류면 diagonal/stripe가 강하게 나옴)
diag = np.diag(A)
plt.figure(figsize=(10,3))
plt.plot(diag); plt.title("Diagonal trace"); plt.tight_layout()
plt.savefig("diag_trace.png", dpi=180); plt.close()

# (3) 축 교환했을 때 히트맵 비교
def save_hm(M, name):
    lo, hi = np.percentile(M, [1,99])
    plt.figure(figsize=(6,4))
    plt.imshow(np.clip(M, lo, hi), origin="lower", aspect="auto")
    plt.title(name); plt.colorbar(); plt.tight_layout()
    plt.savefig(f"{name}.png", dpi=180); plt.close()

save_hm(A, "hm_raw")
save_hm(A.T, "hm_raw_T")

print("saved: axis_mean_compare.png, diag_trace.png, hm_raw.png, hm_raw_T.png")
