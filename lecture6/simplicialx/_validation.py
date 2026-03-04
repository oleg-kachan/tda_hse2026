import numpy as np


# ==== FUNCTIONS FOR CHECKING INPUT CORRECTNESS ====


def _check_kind(kind):
    if kind not in ['clique', 'neighborhood', 'dowker']:
        raise ValueError('Incorrect complex type; can be "clique", "neighborhood" or "dowker"')


def _check_simplex(simplex):
    if not isinstance(simplex, (np.ndarray)) or simplex.ndim != 1 or simplex.shape[0] == 0:
        t = str(type(simplex))
        raise TypeError('Incorrect type of simplex: given ' + t + '; must be non-empty 1d numpy.ndarray')
    if len(set(simplex)) != simplex.shape[0]:
        raise ValueError('Incorrect values in simplex; each simplex must contain non-repeating integer values')


def _check_simplex_in_complex(simplices, simplex):
    if not simplex.tolist() in simplices.tolist():
        raise ValueError('Simplex ' + str(simplex) + ' is not found in this simplicial complex')


def _check_simplices(simplices):
    for s in simplices:
        _check_simplex(s)


def _check_n_simpl_complex(n_complex):
    if not isinstance(n_complex, list):
        raise TypeError('Incorrect type of n_complex: must be list of np.ndarrays of simplices')
    for i in range(len(n_complex)):
        if not isinstance(n_complex[i], np.ndarray):
            raise TypeError('Incorrect type of n_complex: must be list of np.ndarrays of simplices')
        for simplex in n_complex[i]:
            _check_simplex(simplex)


def _check_adjacency_matrix(X):
    if not isinstance(X, (np.ndarray)) or X.ndim != 2 or X.shape[0] != X.shape[1]:
        raise TypeError('Incorrect type of adjacency matrix X: must be numpy.ndarray of shape V x V')


def _check_weights(weights):
    if not isinstance(weights, dict):
        raise TypeError('Incorrect weights type: must be dict with tuple keys (simplices) and int or float values')
    for key in weights.keys():
        if not isinstance(key, tuple):
            raise TypeError('Incorrect type of dictionary key: must be tuple')
        if not isinstance(weights[key], (float, int)):
            raise TypeError('Incorrect type of dictionary value: must be int or float')


def _check_integer_values(**kwarg):
    out = []
    for name, val in kwarg.items():
        try:
            out.append(int(val))
        except:
            raise TypeError('Incorrect type of ' + name + ': must be integer')

    if len(out) > 1:
        return tuple(out)
    else:
        return out[0]


def _check_float_values(**kwarg):
    out = []
    for name, val in kwarg.items():
        try:
            out.append(float(val))
        except:
            raise TypeError('Incorrect type of ' + name + ': must be float')

    if len(out) > 1:
        return tuple(out)
    else:
        return out[0]


def _check_for_laplace(k, p, q, orient):
    k, p, q = _check_integer_values(k=k, p=p, q=q)
    if k < 0:
        raise ValueError('Incorrect value of k: must be >= 0')
    if p < 1:
        raise ValueError('Incorrect value of p: must be >= 1')
    if k - p < -1:
        raise ValueError('Incorrect value of k and p: k - p must be >= -1')
    if q < 1:
        raise ValueError('Incorrect value of q: must be >= 1')
    if orient not in [-1, 1]:
        raise ValueError('Incorrect value of orient: must be 1 or -1')
    return k, p, q
