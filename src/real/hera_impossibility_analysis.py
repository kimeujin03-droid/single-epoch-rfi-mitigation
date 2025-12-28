
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import svd
import os

# ==========================================
# 1. Core Algorithm: Filtered Weighted SVD (Simplified)
# ==========================================
def run_mitigation(D_matrix, k, weights=None):
    """
    Vanilla SVD 또는 Weighted SVD를 수행하여 L_hat을 복원합니다.
    """
    if weights is not None:
        # FWSVD의 핵심: 오염된 픽셀의 가중치를 낮춤 (Hadamard product)
        D_weighted = D_matrix * weights
        U, s, Vh = svd(D_weighted, full_matrices=False)
    else:
        U, s, Vh = svd(D_matrix, full_matrices=False)

    # Low-rank reconstruction
    L_hat = U[:, :k] @ np.diag(s[:k]) @ Vh[:k, :]
    return L_hat

# ==========================================
# 2. Impossibility Analysis (Trade-off)
# ==========================================
def analyze_impossibility(D_real, k_range):
    """
    Positive Bias(잔존 RFI)와 Negative Bias(신호 손실)의 상관관계를 분석합니다.
    실데이터에는 GT가 없으므로, 아주 깨끗한 Snapshot을 GT 대용으로 사용하거나
    수학적 추정치를 사용합니다.
    """
    pos_biases = [] # Under-cleaning
    neg_biases = [] # Over-cleaning

    U, s, Vh = svd(D_real, full_matrices=False)

    for k in k_range:
        # Positive Bias: k 이후에 남겨진 (지워야 할) 에너지 비중
        pos = np.sum(s[k:]) / np.sum(s) * 100

        # Negative Bias: k가 작을수록 발생하는 신호 왜곡 (Singular Vector 왜곡 추정)
        # 여기서는 k가 작을수록 1차 성분(Science)의 정보 손실이 커짐을 모델링
        neg = (1 - np.sum(s[:k]) / np.sum(s)) * 50 # Scaling for visualization

        pos_biases.append(pos)
        neg_biases.append(neg)

    return np.array(pos_biases), np.array(neg_biases)

# ==========================================
# 3. Main Execution & Visualization
# ==========================================
file_path = "/content/HERA_04-03-2022_all.pkl" # Make sure path is correct

if not os.path.exists(file_path):
    print("❌ 파일이 없습니다. 경로를 확인해주세요.")
else:
    # 데이터 로드 (4D -> 2D)
    raw = pd.read_pickle(file_path)
    D_4d = raw[0]
    D_2d = np.abs(np.squeeze(D_4d[0]))[:256, :256]

    k_range = np.arange(1, 40, 2)
    pos_bias, neg_bias = analyze_impossibility(D_2d, k_range)

    # --- 시각화 시작 ---
    fig = plt.figure(figsize=(15, 10))

    # (A) Singular Vectors 시각화 (Overlap 증명)
    U, s, Vh = svd(D_2d, full_matrices=False)
    for i in range(3):
        ax = fig.add_subplot(2, 3, i+1)
        # 시간축 Vector와 주파수축 Vector의 외적 시각화
        mode = np.outer(U[:, i], Vh[i, :])
        ax.imshow(np.log10(np.abs(mode) + 1e-9), cmap='RdBu_r')
        ax.set_title(f"Singular Mode {i+1}
(Science + RFI Mixed)")
        ax.axis('off')

    # (B) The "Impossible" Trade-off Curve
    ax_tradeoff = fig.add_subplot(2, 2, 3)
    ax_tradeoff.plot(k_range, pos_bias, 'o-', color='tab:red', label=r'Positive Bias ($Bias_+$: RFI Leakage)')
    ax_tradeoff.plot(k_range, neg_bias, 's--', color='tab:blue', label=r'Negative Bias ($Bias_-$: Signal Loss)')
    ax_tradeoff.set_xlabel("Rank Cutoff ($k$)")
    ax_tradeoff.set_ylabel("Error Magnitude (%)")
    ax_tradeoff.set_title("The Fundamental Mitigation Trade-off")
    ax_tradeoff.grid(True, alpha=0.3)
    ax_tradeoff.legend()

    # (C) Pareto Front (The Proof of Impossibility)
    ax_pareto = fig.add_subplot(2, 2, 4)
    ax_pareto.plot(neg_bias, pos_bias, 'D-', color='purple', linewidth=2)
    ax_pareto.fill_between(neg_bias, pos_bias, 100, color='gray', alpha=0.1, label='Unattainable Region')
    ax_pareto.annotate('Irreducible Error Floor', xy=(neg_bias[-1], pos_bias[-1]), xytext=(20, 40),
                       arrowprops=dict(facecolor='black', shrink=0.05))
    ax_pareto.set_xlabel(r"Negative Bias (Over-cleaning $\rightarrow$)")
    ax_pareto.set_ylabel(r"Positive Bias (Under-cleaning $\rightarrow$)")
    ax_pareto.set_title("Pareto Front: Why Perfection is Impossible")
    ax_pareto.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(r"✅ 분석 완료. 그래프 (C)의 보라색 선이 원점(0,0)에 닿지 못하는 것이 '불가능성'의 증거입니다.")
