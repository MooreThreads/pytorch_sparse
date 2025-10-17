import importlib
import os.path as osp

import torch

__version__ = '0.6.18'

for library in [
        '_version', '_convert', '_diag', '_spmm', '_metis', '_rw', '_saint',
        '_sample', '_ego_sample', '_hgt_sample', '_neighbor_sample', '_relabel'
]:
    # musa_spec = importlib.machinery.PathFinder().find_spec(
    #     f'{library}_musa', [osp.dirname(__file__)])
    # cpu_spec = importlib.machinery.PathFinder().find_spec(
    #     f'{library}_cpu', [osp.dirname(__file__)])
    # spec = musa_spec or cpu_spec
    spec = importlib.machinery.PathFinder().find_spec(
        f'{library}', [osp.dirname(__file__)])
    if spec is not None:
        torch.ops.load_library(spec.origin)
    else:  # pragma: no cover
        raise ImportError(f"Could not find module '{library}_cpu' in "
                          f"{osp.dirname(__file__)}")

musa_version = torch.ops.torch_sparse.musa_version()
if torch.version.musa is not None and musa_version != -1:  # pragma: no cover
    if str(musa_version) != torch.version.musa:
        raise RuntimeError(
            f'Detected that PyTorch and torch_sparse were compiled with '
            f'different MUSA versions. PyTorch has MUSA version '
            f'{torch.version.musa} and torch_sparse has MUSA version '
            f'{musa_version}. Please reinstall the torch_sparse that '
            f'matches your PyTorch install.')

from .storage import SparseStorage  # noqa
from .tensor import SparseTensor  # noqa
from .transpose import t  # noqa
from .narrow import narrow, __narrow_diag__  # noqa
from .select import select  # noqa
from .index_select import index_select, index_select_nnz  # noqa
from .masked_select import masked_select, masked_select_nnz  # noqa
from .permute import permute  # noqa
from .diag import remove_diag, set_diag, fill_diag, get_diag  # noqa
from .add import add, add_, add_nnz, add_nnz_  # noqa
from .mul import mul, mul_, mul_nnz, mul_nnz_  # noqa
from .reduce import sum, mean, min, max  # noqa
from .matmul import matmul  # noqa
from .cat import cat  # noqa
from .rw import random_walk  # noqa
from .metis import partition  # noqa
from .bandwidth import reverse_cuthill_mckee  # noqa
from .saint import saint_subgraph  # noqa
from .sample import sample, sample_adj  # noqa

from .convert import to_torch_sparse, from_torch_sparse  # noqa
from .convert import to_scipy, from_scipy  # noqa
from .coalesce import coalesce  # noqa
from .transpose import transpose  # noqa
from .eye import eye  # noqa
from .spmm import spmm  # noqa
from .spspmm import spspmm  # noqa
from .spadd import spadd  # noqa

__all__ = [
    'SparseStorage',
    'SparseTensor',
    't',
    'narrow',
    '__narrow_diag__',
    'select',
    'index_select',
    'index_select_nnz',
    'masked_select',
    'masked_select_nnz',
    'permute',
    'remove_diag',
    'set_diag',
    'fill_diag',
    'get_diag',
    'add',
    'add_',
    'add_nnz',
    'add_nnz_',
    'mul',
    'mul_',
    'mul_nnz',
    'mul_nnz_',
    'sum',
    'mean',
    'min',
    'max',
    'matmul',
    'cat',
    'random_walk',
    'partition',
    'reverse_cuthill_mckee',
    'saint_subgraph',
    'to_torch_sparse',
    'from_torch_sparse',
    'to_scipy',
    'from_scipy',
    'coalesce',
    'transpose',
    'eye',
    'spmm',
    'spspmm',
    'spadd',
    '__version__',
]
