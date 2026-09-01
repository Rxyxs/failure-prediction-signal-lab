"""Estrategia de validacion cruzada con GroupKFold para evitar fuga temporal.

Los segmentos consecutivos de una misma senal continua estan correlacionados
temporalmente. Para evitar fuga de informacion entre entrenamiento y
validacion, se agrupan los segmentos en bloques contiguos (``group_id``) y se
aplica ``GroupKFold`` sobre esos grupos, garantizando que ningun grupo
aparezca simultaneamente en train y en validation.

Autor: Pablo Reyes
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold


def make_temporal_groups(n_samples: int, n_groups: int) -> np.ndarray:
    """Asigna cada segmento a un grupo contiguo, preservando el orden temporal."""
    group_size = max(1, n_samples // n_groups)
    groups = np.minimum(np.arange(n_samples) // group_size, n_groups - 1)
    return groups


def get_group_kfold_splits(n_samples: int, n_splits: int = 5):
    """Genera los indices de train/validation usando GroupKFold sobre grupos temporales."""
    groups = make_temporal_groups(n_samples, n_groups=n_splits * 2)
    gkf = GroupKFold(n_splits=n_splits)
    indices = np.arange(n_samples)
    return list(gkf.split(indices, groups=groups)), groups
