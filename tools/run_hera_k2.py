import os
import importlib.util

p = os.path.join(os.path.dirname(__file__), 'make_figure_svd_fws_tsvd_panels.py')
spec = importlib.util.spec_from_file_location('mk', p)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

outpath = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'figure_svd_fws_tsvd_hera_k2.png')
print(f"Calling run_hera_eor_injection(outpath={outpath}, rank=2)")
mod.run_hera_eor_injection(outpath=outpath, sample_idx=0, rank=2)
print('done')
