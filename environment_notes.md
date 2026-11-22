# Environment notes
- Conda env: `pa1` (Python 3.11.17, torch 2.4.1+cu124). Interpreter used for everything: `/home/huzaifa/miniconda3/envs/pa1/bin/python`.
- Packages that were MISSING in `pa1` and installed with `python -m pip install` (note: use `python -m pip`, bare `pip` may point at the system Python):
  pandas, h5py, tables, matplotlib, scikit-learn, jupyter, ipykernel, nbconvert, nbclient, nbformat, tqdm, kaggle, reportlab, psutil, py-cpuinfo, networkx, seaborn.
- Already present in `pa1`: torch 2.4.1+cu124, numpy, scipy, markdown-it-py, pillow.
- Jupyter kernel registered as `pa1`: `python -m ipykernel install --user --name pa1 --display-name pa1`.
- Exact versions: see `requirements.txt` (generated with `pip list --format=freeze`).
