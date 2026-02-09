#!/usr/bin/env python3
"""
Load HERA_04-03-2022_all.pkl, try to find proxy and true arrays, compute
Spearman/Pearson correlation, and save a scatter plot.
"""
from __future__ import annotations

import pickle
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

try:
    from scipy.stats import spearmanr, pearsonr
except Exception:
    spearmanr = None
    pearsonr = None

PKL = 'HERA_04-03-2022_all.pkl'
OUTPNG = 'figure_proxy_correlation_from_pkl.png'
OUTPNG_CLEAN = 'figure_proxy_correlation_from_pkl_clean95.png'

def is_numeric_array(v):
    try:
        a = np.asarray(v)
    except Exception:
        return False
    if not np.issubdtype(a.dtype, np.number):
        return False
    if a.size == 0:
        return False
    # accept 1D or 2D where one dim is 1
    if a.ndim == 1:
        return True
    if a.ndim == 2 and (a.shape[0] == 1 or a.shape[1] == 1):
        return True
    return False


def flatten_numeric(v):
    a = np.asarray(v)
    if a.ndim == 1:
        return a.astype(float)
    if a.ndim == 2:
        if a.shape[0] == 1:
            return a.flatten().astype(float)
        if a.shape[1] == 1:
            return a.flatten().astype(float)
    # try ravel
    return a.ravel().astype(float)


def find_candidates_from_dict(d):
    cand = {}
    for k, v in d.items():
        if is_numeric_array(v):
            arr = flatten_numeric(v)
            cand[k] = arr
    return cand


def pick_key(cands, wants):
    # exact key match
    keys = list(cands.keys())
    for w in wants:
        if w in keys:
            return w
    # substring match
    lower_map = {k: k.lower() for k in keys}
    for w in wants:
        for k, kl in lower_map.items():
            if w in kl:
                return k
    # fallback: first key
    if keys:
        return keys[0]
    return None


def main():
    if not os.path.exists(PKL):
        print(f"Pickle file not found: {PKL}", file=sys.stderr)
        sys.exit(2)

    with open(PKL, 'rb') as f:
        obj = pickle.load(f)

    print('Loaded pickle of type', type(obj))

    candidates = {}
    # Special handling for list-like pickles containing data + masks (HERA case)
    if isinstance(obj, list):
        # collect numeric arrays and boolean masks
        data_arrays = {}
        mask_arrays = {}
        for i, el in enumerate(obj):
            try:
                a = np.asarray(el)
            except Exception:
                continue
            if a.size == 0:
                continue
            if a.dtype == bool:
                mask_arrays[f'idx_{i}'] = a.squeeze()
            elif np.issubdtype(a.dtype, np.number):
                data_arrays[f'idx_{i}'] = a.squeeze()
        print('Found data arrays:', list(data_arrays.keys()), 'mask arrays:', list(mask_arrays.keys()))
        # try to find a data/mask pair with matching leading dimension
        paired = False
        for dk, da in data_arrays.items():
            for mk, ma in mask_arrays.items():
                if da.ndim >= 1 and ma.ndim >= 1 and da.shape[0] == ma.shape[0]:
                    # compute per-sample metrics
                    N = da.shape[0]
                    total_power = np.zeros(N, dtype=float)
                    flagged_power = np.zeros(N, dtype=float)
                    rfi_configs = [None] * N
                    for t in range(N):
                        dt = da[t]
                        mskt = ma[t]
                        # ensure shapes broadcastable
                        try:
                            # compute squared magnitude
                            arr = np.asarray(dt, dtype=float)
                            total_power[t] = float((arr * arr).sum())
                            if np.any(mskt):
                                flagged = arr[mskt]
                                flagged_power[t] = float((flagged * flagged).sum())
                                # record which indices were flagged for this sample
                                try:
                                    idxs = np.where(mskt)[0]
                                    # store as tuple for hashability
                                    rfi_configs[t] = tuple(int(x) for x in idxs.tolist())
                                except Exception:
                                    rfi_configs[t] = tuple()
                            else:
                                flagged_power[t] = 0.0
                                rfi_configs[t] = tuple()
                        except Exception:
                            total_power[t] = 0.0
                            flagged_power[t] = 0.0
                    eps = 1e-16
                    proxy = 100.0 * (flagged_power / (total_power + eps))
                    # normalize flagged_power to percent of its max to keep scales similar
                    true = 100.0 * (flagged_power / (flagged_power.max() + eps))
                    # build a DataFrame for inspection
                    df = pd.DataFrame({
                        'sample_idx': np.arange(N),
                        'computed_proxy': proxy,
                        'true_rfi_leakage': true,
                        'total_power': total_power,
                        'flagged_power': flagged_power,
                        'rfi_config': rfi_configs,
                        'num_flagged': [len(x) if x is not None else 0 for x in rfi_configs]
                    })
                    # make rfi_config a string for nicer printing
                    df['rfi_config_str'] = df['rfi_config'].apply(lambda x: ','.join(map(str, x)) if (x is not None and len(x)>0) else '')
                    print(f'Using data {dk} and mask {mk} to build proxy/true arrays (length {N})')
                    paired = True
                    break
            if paired:
                break
        if not paired:
            print('Could not form proxy/true arrays from list-structured pickle; falling back to dict-like search')
            # fall back to dict behavior below
            obj_dict = None
    if isinstance(obj, dict):
        candidates = find_candidates_from_dict(obj)
        print('Found numeric candidates in pickle dict:', list(candidates.keys()))
    else:
        # if it's e.g. a DataFrame-like object
        try:
            if hasattr(obj, 'columns') and hasattr(obj, 'iloc'):
                df = obj
                for c in df.columns:
                    if np.issubdtype(df[c].dtype, np.number):
                        candidates[c] = df[c].values.astype(float)
                print('Loaded DataFrame-like object from pickle, columns:', list(candidates.keys()))
            else:
                # try to coerce to array
                if is_numeric_array(obj):
                    candidates['data'] = flatten_numeric(obj)
                    print('Pickle holds a numeric array of length', candidates['data'].size)
        except Exception:
            pass

    # If proxy/true not already computed above, select from candidates
    if not ('proxy' in locals() and 'true' in locals()):
        if not candidates:
            print('No numeric arrays found in pickle to use as proxy/true.', file=sys.stderr)
            sys.exit(3)

        # preferred keys for proxy/true
        proxy_wants = ['proxy', 'svd', 'leak', 'bias', 'proxy_bias', 'svd_mean', 'svd_median', 'rfi']
        true_wants = ['true', 'injected', 'error', 'true_bias', 'seed']

        proxy_key = pick_key(candidates, proxy_wants)
        true_key = pick_key(candidates, true_wants)

        # if same, pick alternate
        if proxy_key == true_key:
            for k in candidates.keys():
                if k != proxy_key:
                    true_key = k
                    break

        if proxy_key not in candidates or true_key not in candidates:
            # as final fallback, pick two numeric arrays of same length
            lengths = {k: v.size for k, v in candidates.items()}
            pairs = [(a, b) for a in candidates for b in candidates if a != b and candidates[a].size == candidates[b].size]
            if pairs:
                proxy_key, true_key = pairs[0]
                print('Fallback selected pair:', proxy_key, true_key)
            else:
                print('Could not find a pair of numeric arrays of matching length in the pickle.', file=sys.stderr)
                print('Candidates and lengths:', lengths)
                sys.exit(4)

        proxy = candidates[proxy_key].astype(float)
        true = candidates[true_key].astype(float)

        print(f'Using proxy key: {proxy_key} (len={proxy.size}); true key: {true_key} (len={true.size})')
    else:
        proxy_key = 'computed_proxy'
        true_key = 'computed_true'

    # If we built a DataFrame `df` earlier, perform the problematic-sample inspection
    if 'df' in locals():
        mask = df['computed_proxy'] > 99.0
        problematic = df[mask]
        print('\nNumber of problematic samples (proxy>99):', len(problematic))
        if len(problematic) > 0:
            unique_configs = problematic['rfi_config_str'].unique()
            print('\nUnique rfi_config strings for problematic samples:')
            for uc in unique_configs:
                print('  ', repr(uc))
            print('\ntrue_rfi_leakage description for problematic samples:')
            print(problematic['true_rfi_leakage'].describe())

    # ensure same length
    n = min(proxy.size, true.size)
    proxy = proxy[:n]
    true = true[:n]

    # If we built a DataFrame above, allow plotting a "clean" subset
    # where saturation is removed (computed_proxy < 95). Otherwise
    # proceed with the full arrays computed/found above.
    plot_variants = []

    # full dataset (as before)
    plot_variants.append(('full', proxy, true, OUTPNG))

    # if we have df built from the data+mask pairing, make a filtered/clean version
    if 'df' in locals():
        clean_df = df[df['computed_proxy'] < 95.0]
        if len(clean_df) == 0:
            print('Clean subset (proxy<95) is empty; skipping clean plot.')
        else:
            proxy_clean = clean_df['computed_proxy'].values.astype(float)
            true_clean = clean_df['true_rfi_leakage'].values.astype(float)
            plot_variants.append(('clean95', proxy_clean, true_clean, OUTPNG_CLEAN))

    # compute correlations and write plots for each variant
    for name, px, tx, outpng in plot_variants:
        if px.size == 0 or tx.size == 0:
            print(f'Skipping plot {name}: empty arrays')
            continue
        if spearmanr is not None:
            rho_s, p_s = spearmanr(px, tx)
        else:
            rho_s, p_s = (float('nan'), float('nan'))
        if pearsonr is not None:
            rho_p, p_p = pearsonr(px, tx)
        else:
            rho_p = float(np.corrcoef(px, tx)[0, 1])
            p_p = float('nan')

        print(f'[{name}] Spearman rho={rho_s:.3f}, Pearson r={rho_p:.3f} (n={px.size})')

        plt.figure(figsize=(5.5, 5))
        plt.scatter(px, tx, s=28, alpha=0.7)
        xmin = min(float(px.min()), float(tx.min()))
        xmax = max(float(px.max()), float(tx.max()))
        plt.plot([xmin, xmax], [xmin, xmax], 'k--', lw=1, alpha=0.6)
        plt.xlabel(f'Proxy ({proxy_key})')
        plt.ylabel(f'True ({true_key})')
        plt.title(f'Proxy vs True from {PKL} ({name})\nSpearman {rho_s:.2f}, Pearson {rho_p:.2f}')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(outpng, dpi=200)
        print('WROTE', outpng)

if __name__ == '__main__':
    main()
