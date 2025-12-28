from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from astropy.io import fits

@dataclass
class RealData:
    """
    관측 데이터를 담는 표준 데이터 클래스
    D: (Time, Frequency) 형태의 2D 행렬 (Power)
    time: (T,) 형태의 시간축 (MJD 또는 초 단위)
    freq: (F,) 형태의 주파수축 (MHz 단위)
    meta: 로딩 정보 및 파라미터를 담은 딕셔너리
    """
    D: np.ndarray
    time: Optional[np.ndarray]
    freq: Optional[np.ndarray]
    meta: Dict[str, Any]

# --- Internal Helpers ---

def _ms_get_spw_id_and_chanfreq(ms_path: str, prefer_spw: Optional[int] = None) -> Tuple[int, np.ndarray]:
    """
    MeasurementSet(MS)에서 Spectral Window ID와 해당 채널의 주파수(Hz)를 추출합니다.
    """
    from casacore.tables import table

    # 1) Spectral Window ID 결정
    if prefer_spw is not None:
        spw_id = int(prefer_spw)
    else:
        # DATA_DESCRIPTION 테이블을 참조하여 첫 번째 SPW ID를 가져옴
        dd = table(f"{ms_path}/DATA_DESCRIPTION", readonly=True, ack=False)
        try:
            spw_ids = dd.getcol("SPECTRAL_WINDOW_ID")
            if spw_ids.size == 0:
                raise ValueError("DATA_DESCRIPTION 테이블에 SPECTRAL_WINDOW_ID가 없습니다.")
            spw_id = int(np.unique(spw_ids)[0])
        finally:
            dd.close()

    # 2) 해당 SPW의 주파수 축(CHAN_FREQ) 읽기
    spw_tab = table(f"{ms_path}/SPECTRAL_WINDOW", readonly=True, ack=False)
    try:
        chan_freq = spw_tab.getcol("CHAN_FREQ") # 보통 (nspw, nchan) 또는 (nchan, nspw)
        arr = np.array(chan_freq)

        if arr.ndim != 2:
            raise ValueError(f"CHAN_FREQ의 차원이 예상과 다릅니다: {arr.shape}")

        # 차원 방향 확인 및 인덱싱
        if arr.shape[1] > spw_id and arr.shape[0] >= 1:
            freq_hz = arr[:, spw_id] # (nchan, nspw) 케이스
        elif arr.shape[0] > spw_id and arr.shape[1] >= 1:
            freq_hz = arr[spw_id, :] # (nspw, nchan) 케이스
        else:
            raise ValueError(f"spw_id={spw_id}를 인덱싱할 수 없습니다. Shape: {arr.shape}")

        freq_hz = np.array(freq_hz, dtype=np.float64, copy=False)
        return spw_id, np.squeeze(freq_hz)
    finally:
        spw_tab.close()

def _ensure_increasing(freq_hz: np.ndarray, D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    주파수 축이 오름차순인지 확인하고, 내림차순일 경우 데이터와 함께 뒤집습니다.
    비단조(non-monotonic)일 경우 정렬합니다.
    """
    if freq_hz.size != D.shape[1]:
        return freq_hz, D

    # 이미 오름차순인 경우
    if np.all(np.diff(freq_hz) > 0):
        return freq_hz, D
    
    # 내림차순인 경우 (단순 반전으로 성능 최적화)
    if np.all(np.diff(freq_hz) < 0):
        return freq_hz[::-1].copy(), D[:, ::-1].copy()
    
    # 그 외의 경우 (정렬 필요)
    idx = np.argsort(freq_hz)
    return freq_hz[idx].copy(), D[:, idx].copy()

# --- Main Loaders ---

def load_ms_to_tf(
    path: str,
    pol_idx: int = 0,
    amp_mode: str = "abs2",
    prefer_spw: Optional[int] = None,
    require_single_spw: bool = True,
) -> RealData:
    """
    MeasurementSet 데이터를 Time-Frequency(Dynamic Spectrum) 평면으로 변환합니다.
    - 여러 베이스라인의 가시도(Visibility)를 동일 시간대별로 평균(Averaging)합니다.
    - Flag 처리된 데이터는 NaN으로 변환합니다.
    """
    try:
        from casacore.tables import table
    except ImportError:
        raise ImportError("CASA 데이터를 읽으려면 python-casacore가 필요합니다. 'pip install python-casacore'를 실행하세요.")

    # 1. 주파수 정보 획득
    spw_id, freq_hz = _ms_get_spw_id_and_chanfreq(path, prefer_spw=prefer_spw)
    
    tb = table(path, readonly=True, ack=False)
    try:
        # 2. SPW 일관성 검사 (옵션)
        if require_single_spw:
            dd_tab = table(f"{path}/DATA_DESCRIPTION", readonly=True, ack=False)
            try:
                dd_spw = dd_tab.getcol("SPECTRAL_WINDOW_ID")
                data_desc_id = tb.getcol("DATA_DESC_ID")
                row_spw = dd_spw[data_desc_id]
                uniq_row_spw = np.unique(row_spw)
                if uniq_row_spw.size > 1:
                    raise ValueError(f"MS에 여러 SPW가 섞여 있습니다: {uniq_row_spw}. 필터링이 필요합니다.")
            finally:
                dd_tab.close()

        # 3. 메인 데이터 로드
        data = tb.getcol("DATA")   # (nrow, nchan, npol)
        time = tb.getcol("TIME")   # (nrow,)
        flag = tb.getcol("FLAG")   # (nrow, nchan, npol)

        if pol_idx >= data.shape[2]:
            raise ValueError(f"pol_idx({pol_idx})가 데이터의 편파 수({data.shape[2]})를 초과합니다.")

        # 4. 가시도 계산 및 Flag 적용
        vis = np.where(flag[:, :, pol_idx], np.nan, data[:, :, pol_idx])
        
        if amp_mode == "abs2":
            p = (vis.real**2 + vis.imag**2)
        elif amp_mode == "abs":
            p = np.abs(vis)
        else:
            raise ValueError("amp_mode는 'abs' 또는 'abs2'여야 합니다.")

        # 5. 시간 축 기준 베이스라인 평균 (Dynamic Spectrum 생성)
        uniq_t = np.unique(time)
        T, F = len(uniq_t), data.shape[1]
        D = np.full((T, F), np.nan, dtype=np.float32)

        for i, t in enumerate(uniq_t):
            mask = (time == t)
            # nanmean을 사용하여 특정 베이스라인만 플래깅된 경우에도 대응
            D[i] = np.nanmean(p[mask, :], axis=0).astype(np.float32)

        # 6. 주파수 축 정렬 및 MHz 변환
        freq_hz_sorted, D_sorted = _ensure_increasing(freq_hz, D)
        freq_mhz = (freq_hz_sorted / 1e6).astype(np.float64)

        meta = {
            "format": "ms",
            "path": path,
            "pol_idx": pol_idx,
            "amp_mode": amp_mode,
            "spw_id": int(spw_id),
            "nrow": int(data.shape[0]),
            "nchan": int(F),
        }

        return RealData(D=D_sorted, time=uniq_t.astype(np.float64), freq=freq_mhz, meta=meta)

    finally:
        tb.close()

def load_dynamic_spectrum_fits(path: str, data_hdu: int = 0) -> RealData:
    """
    이미 2D Dynamic Spectrum 형태로 저장된 FITS 파일을 로드합니다.
    """
    with fits.open(path, memmap=True) as hdul:
        data = np.squeeze(hdul[data_hdu].data)
        if data.ndim != 2:
            raise ValueError(f"FITS 데이터가 2D가 아닙니다. Shape: {data.shape}")
        
        D = data.astype(np.float32, copy=False)
        return RealData(
            D=D, 
            time=None, 
            freq=None, 
            meta={"format": "fits_dynspec", "path": path}
        )

def load_real_data(path: str, fmt: str, **kwargs) -> RealData:
    """
    형식(ms, fits_dynspec)에 따라 데이터를 로드하는 통합 인터페이스입니다.
    """
    if fmt == "ms":
        return load_ms_to_tf(path, **kwargs)
    elif fmt == "fits_dynspec":
        return load_dynamic_spectrum_fits(path, **kwargs)
    else:
        raise ValueError(f"지원하지 않는 포맷입니다: {fmt}")
