
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
import pandas as pd
import os

def analyze_svd_structure(data_matrix, k_cutoff=5, title="HERA Data"):
    """
    D = L + S + E 모델을 SVD로 분해하고 Spectrum을 분석합니다.
    k_cutoff: 과학 신호(Science)가 지배적이라고 판단되는 Rank 임계값
    """
    # 1. SVD 수행
    U, s, Vh = svd(data_matrix, full_matrices=False)

    # 2. L (Low-rank) 재구성: 상위 k개 성분만 사용
    s_clean = s.copy()
    s_clean[k_cutoff:] = 0
    L_hat = U @ np.diag(s_clean) @ Vh

    # 3. S + E (Residual) 재구성
    S_hat = data_matrix - L_hat

    # --- 시각화 ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) Original Spectrogram
    im0 = axes[0, 0].imshow(np.log10(np.abs(data_matrix) + 1e-9), aspect='auto', cmap='viridis')
    axes[0, 0].set_title(f"Original D (Log Scale)
{title}")
    plt.colorbar(im0, ax=axes[0, 0])

    # (2) Singular Value Spectrum (Raw String 적용으로 에러 해결)
    axes[0, 1].semilogy(s, 'o-', label="Original Spectrum", alpha=0.8)
    axes[0, 1].semilogy(s_clean[:k_cutoff+5], 'r--', label="Cleaned (Restricted)", alpha=0.8)
    axes[0, 1].axvline(x=k_cutoff, color='black', linestyle=':', label='Science Cutoff (k)')
    axes[0, 1].set_title(r"Singular Value Spectrum ($\Sigma$)") # r prefix 추가
    axes[0, 1].set_xlabel("Rank Index")
    axes[0, 1].set_ylabel("Singular Value (log)")
    axes[0, 1].legend()

    # (3) Reconstructed L (Science-dominated)
    im2 = axes[1, 0].imshow(np.log10(np.abs(L_hat) + 1e-9), aspect='auto', cmap='magma')
    axes[1, 0].set_title(f"Recovered $L$ (Rank={k_cutoff})")
    plt.colorbar(im2, ax=axes[1, 0])

    # (4) Under-cleaning Ratio 계산
    under_cleaning_ratio = np.sum(s[k_cutoff:]) / np.sum(s) * 100

    axes[1, 1].text(0.1, 0.6, f"Under-cleaning Ratio: {under_cleaning_ratio:.2f}%", fontsize=12, weight='bold')
    axes[1, 1].text(0.1, 0.4, f"Energy in Science Subspace: {np.sum(s[:k_cutoff]):.2f}", fontsize=12)
    axes[1, 1].set_axis_off()
    axes[1, 1].set_title("Quantitative Bias Metrics")

    plt.tight_layout()
    plt.show()


# --- Main Execution ---
file_path = "/content/HERA_04-03-2022_all.pkl" # Make sure path is correct

if not os.path.exists(file_path):
    print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
else:
    try:
        raw_data_list = pd.read_pickle(file_path)
        D_4d = raw_data_list[0]

        print(f"✅ 4D 데이터 포착: {D_4d.shape}")

        sample_2d = np.squeeze(D_4d[0])
        sample_2d = np.abs(sample_2d)

        print(f"✨ 2D 변환 성공! 분석 행렬 크기: {sample_2d.shape}")

        analysis_patch = sample_2d[:256, :256] # Use a patch for faster analysis

        print("💡 SVD 연산 및 시각화 시작...")
        analyze_svd_structure(analysis_patch, k_cutoff=10, title="HERA 2D Slice (Sample 0)")

    except Exception as e:
        print(f"❌ 분석 실패: {e}")
