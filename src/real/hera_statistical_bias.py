
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
import pandas as pd # For loading pickle and DataFrame for results

def calculate_statistical_bias(D_4d, k_range=range(1, 31, 2)):
    """
    420개 전체 샘플에 대해 k값 변화에 따른 Under-cleaning Ratio의 통계를 계산합니다.
    D_4d shape: (420, 512, 512, 1)
    """
    # 1. 데이터 준비 (4D -> 3D로 압축 및 절대값)
    D_3d = np.abs(np.squeeze(D_4d)) # (420, 512, 512)
    num_samples = D_3d.shape[0]

    results = {k: [] for k in k_range}

    print(f"⏳ {num_samples}개 샘플에 대한 통계 분석 시작...")

    # 2. 통계 계산 루프
    for i in range(num_samples):
        # 각 샘플의 SVD 수행
        _, s, _ = svd(D_3d[i], full_matrices=False)
        total_energy = np.sum(s)

        for k in k_range:
            # Under-cleaning Ratio: k 이후의 에너지가 전체에서 차지하는 비중
            ratio = (np.sum(s[k:]) / total_energy) * 100
            results[k].append(ratio)

        if (i + 1) % 50 == 0:
            print(f"   - {i + 1}/{num_samples} 완료")

    # 3. 평균 및 표준편차 계산
    ks = list(results.keys())
    means = [np.mean(results[k]) for k in ks]
    stds = [np.std(results[k]) for k in ks]

    # 4. 시각화 (r prefix 확실히 적용!)
    plt.figure(figsize=(10, 6))

    # r prefix를 사용하여 \p 이스케이프 경고 방지
    plt.errorbar(ks, means, yerr=stds, fmt='o-', capsize=5,
                 color='tab:blue', ecolor='tab:red',
                 label=r'Mean Bias $\pm$ 1$\sigma$')

    plt.title(r"Statistical Bias Distribution (\mathrm{Bias}_{\sigma}$)", fontsize=14)
    plt.xlabel(r"Rank Cutoff ($k$)", fontsize=12)
    plt.ylabel("Under-cleaning Ratio (%)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.show()

    return pd.DataFrame({"k": ks, "mean": means, "std": stds})

# --- Main Execution ---
file_path = "/content/HERA_04-03-2022_all.pkl" # Make sure path is correct
D_4d = pd.read_pickle(file_path)[0] # Load D_4d

statistical_bias_df = calculate_statistical_bias(D_4d)
print("\nStatistical Bias DataFrame:")
print(statistical_bias_df)
