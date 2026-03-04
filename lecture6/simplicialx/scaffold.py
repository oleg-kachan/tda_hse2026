import numpy as np

from ._validation import _check_adjacency_matrix


def _is_subset(lst1, lst2):
    return set(lst1).issubset(set(lst2))


def _contains(row1, row2):
    return _is_subset(row2.nonzero()[0], row1.nonzero()[0])


def _remove_zero_vertices(x):
    idx = x.any(axis=1)
    x = x[:, idx]
    x = x[idx]
    return x


def _is_scaffold(x):
    for i in range(x.shape[0]):
        for j in list(x[i].nonzero()[0]):
            if i != j and _contains(x[j], x[i]):
                return False
    return True


def core_scaffold(x):
    _check_adjacency_matrix(x)

    x_res = x.copy()
    np.fill_diagonal(x_res, 1)

    while not _is_scaffold(x_res):
        # for every row, check if it contains other rows
        for i in range(x_res.shape[0]):
            for j in list(x_res[i].nonzero()[0]):
                if i != j and _contains(x_res[j], x_res[i]):
                    x_res[i, :] = 0
                    x_res[:, i] = 0

        x_res = _remove_zero_vertices(x_res)

    np.fill_diagonal(x_res, 0)
    return x_res
