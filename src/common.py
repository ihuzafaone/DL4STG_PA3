"""Shared utilities: seeding, paths, hardware info."""
import os, random, platform, json, sys
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW = os.path.join(ROOT, 'data', 'raw')
PROC = os.path.join(ROOT, 'data', 'processed')
RES = os.path.join(ROOT, 'results')
FIG = os.path.join(ROOT, 'figures')
CKPT = os.path.join(ROOT, 'checkpoints')
for _d in (PROC, RES, FIG, CKPT):
    os.makedirs(_d, exist_ok=True)

SEED = 42


def set_seed(seed=SEED, deterministic=True):
    """Seed python, numpy, torch (CPU+CUDA) and request deterministic kernels."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def hardware_info():
    """Collect hardware / software specs programmatically."""
    import psutil
    try:
        import cpuinfo
        cpu = cpuinfo.get_cpu_info().get('brand_raw', platform.processor())
    except Exception:
        cpu = platform.processor() or 'unknown'
    info = dict(
        cpu_model=cpu,
        cpu_physical_cores=psutil.cpu_count(logical=False),
        cpu_logical_cores=psutil.cpu_count(logical=True),
        ram_gb=round(psutil.virtual_memory().total / 1024**3, 1),
        os=platform.platform(),
        python=sys.version.split()[0],
        torch=torch.__version__,
        numpy=np.__version__,
        cuda_available=torch.cuda.is_available(),
    )
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, gpu_memory_gb=round(p.total_memory / 1024**3, 1), cuda_version=torch.version.cuda)
    else:
        info['gpu'] = None
    try:
        import pandas, h5py, matplotlib, sklearn
        info.update(pandas=pandas.__version__, h5py=h5py.__version__, matplotlib=matplotlib.__version__, sklearn=sklearn.__version__)
    except Exception:
        pass
    return info


def save_json(obj, path):
    def _d(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1, default=_d)


def load_json(path):
    with open(path) as f:
        return json.load(f)
