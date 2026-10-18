"""Reproducible seeding for training, evaluation, and probe scripts.

Provides a single ``set_seed(seed)`` so every entry point seeds the same RNGs
the same way. Previously each script re-implemented ``torch.manual_seed`` (and
inconsistently — some seeded numpy, most did not seed ``random`` or the CUDA
RNG). Routing through here closes the "★待实现 set_seed()" gap recorded in
``.trellis/spec/experiments/reproducibility.md``.

Determinism caveat (inherent, see ``.trellis/spec/training/numerics.md``):
``cudnn.benchmark = True`` is set because fixed input shapes per epoch make
autotuning safe and faster; it is NOT bit-deterministic on sm_120 (Blackwell).
The seeded parts (data split, model init, DataLoader worker shuffle, probe
noise generators) ARE deterministic; conv kernel selection is not. For research
code this is acceptable — fold-aggregate metrics vary <1% across reruns.
"""

import os
import random


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed every RNG used by the project.

    Seeds Python ``random``, NumPy, and PyTorch (CPU + all CUDA devices), and
    sets ``cudnn.benchmark = True`` (fast, fixed-shape autotuning). Imports
    torch/numpy lazily so the helper works even in environments that only have
    one of them.

    Parameters
    ----------
    seed : int
        The seed value (default convention across the project is 42, set in
        ``training/cli.py``).
    deterministic : bool, default False
        If True, also enable ``torch.use_deterministic_algorithms`` and set
        ``cudnn.benchmark = False``. Slower; use only when bit-reproducibility
        is required and the platform supports it (note: broken on sm_120, see
        ``training/numerics.md``).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    else:
        # Fixed input shapes per epoch → autotuning is safe and faster.
        torch.backends.cudnn.benchmark = True
