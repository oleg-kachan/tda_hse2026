"""
Current code has some techincal limitations, which I believe can be
overcome. In particular, all orbits suppose that marked_simplex==[0]
and the code can only work with complexes, whose nodes are numbered from 0 to
n-1 for some n.

One can also try to generalise code such that marked_simplex can have higher
dimension, not necessary dim=0. This would lead to orbits for edges, triangles
and etc.
"""

import itertools
import typing as tp
import functools
from dataclasses import dataclass

import numpy as np
import networkx as nx

from .simplicial import SimplicialComplex


##################### Cached Orbits #####################


CACHED_ORBITS_3 = [
    ((0, 1, 2),),
    ((0, 1), (1, 2), (0, 2)),
    ((0, 1), (0, 2)),
    ((0, 1), (1, 2)),
]


CACHED_ORBITS_4 = [
    ((0, 1, 2, 3),),
    ((0, 2, 3), (0, 1, 2), (0, 1, 3), (1, 2, 3)),
    ((0, 2, 3), (0, 1, 2), (0, 1, 3)),
    ((0, 1, 2), (0, 1, 3), (1, 2, 3)),
    ((0, 1), (0, 2, 3), (1, 2, 3)),
    ((1, 2), (0, 1, 3), (0, 2, 3)),
    ((0, 1, 2), (0, 1, 3)),
    ((0, 1, 2), (1, 2, 3)),
    ((0, 1), (0, 2), (0, 3), (1, 2, 3)),
    ((0, 1), (1, 2), (0, 2, 3), (1, 3)),
    ((0, 1), (0, 2), (1, 2, 3)),
    ((0, 1), (1, 2), (0, 2, 3)),
    ((1, 3), (1, 2), (0, 2, 3)),
    ((0, 1), (2, 3), (0, 2), (1, 2), (0, 3), (1, 3)),
    ((0, 1), (0, 2, 3)),
    ((0, 1), (1, 2, 3)),
    ((1, 2), (0, 1, 3)),
    ((0, 1), (0, 2), (1, 2), (0, 3), (1, 3)),
    ((0, 1), (2, 3), (0, 2), (1, 2), (1, 3)),
    ((0, 1), (0, 2), (1, 2), (0, 3)),
    ((0, 1), (0, 2), (1, 2), (1, 3)),
    ((0, 1), (2, 3), (0, 2), (1, 3)),
    ((0, 1), (1, 3), (1, 2), (2, 3)),
    ((0, 1), (0, 2), (0, 3)),
    ((0, 1), (0, 2), (1, 3)),
    ((0, 1), (1, 3), (1, 2)),
    ((0, 1), (1, 2), (2, 3)),
]


##################### Primitives #####################

Extension = tuple[tuple[int, ...], ...]


class SimplicialComplexWrapper(SimplicialComplex):
    def __init__(
            self,
            sc: SimplicialComplex,
            check_nodes: bool = True
    ) -> None:
        if check_nodes:
            assert (
                sc.simplices[0].flatten()
                ==
                np.arange(sc.simplices[0].shape[0])
            ).all(), sc.simplices[0].flatten()

        self._sc = sc
        self._simplices_set: set[tuple[int, ...]] | None = None
        self._adj: list[set[int]] | None = None

    def _compute_simplices_set(self) -> set[tuple[int, ...]]:
        simplices_set: set[tuple[int, ...]] = set()
        for simplex in itertools.chain(*self._sc.simplices):
            simplices_set.add(tuple(sorted(simplex.tolist())))
        return simplices_set
    
    def _compute_adj(self) -> list[set[int]]:
        adj: list[set[int]] = [
            set() for _ in range(self._sc.simplices[0].shape[0])
        ]

        if len(self._sc.simplices) < 2:
            return adj

        for u, v in self._sc.simplices[1]:
            adj[u].add(v)
            adj[v].add(u)

        return adj

    def simplices_set(self) -> set[tuple[int, ...]]:
        if self._simplices_set is None:
            self._simplices_set = self._compute_simplices_set()
        return self._simplices_set
    
    def adj(self, node: int) -> set[int]:
        if self._adj is None:
            self._adj = self._compute_adj()
        return self._adj[node]
    
    @functools.cache
    def compute_num_common_neighbors(self, extension: Extension) -> int:
        count = 0

        extension_nodes = list(set([s[0] for s in extension]))
        candidates = self.adj(extension_nodes[0])
        for node in extension_nodes:
            candidates = candidates & self.adj(node)

        for node in candidates:
            is_common_neighbor = True
            for subset in extension:
                simplex = tuple(sorted([node] + list(subset)))
                if simplex not in self.simplices_set():
                    is_common_neighbor = False
                    break
            if is_common_neighbor:
                count += 1

        return count

    def __getattr__(self, attr):
        return getattr(self._sc, attr)


class Orbit:
    def __init__(
            self,
            sc: SimplicialComplexWrapper,
            marked_simplex: tuple[int, ...],
    ) -> None:
        self._sc = sc
        self._marked_simplex = tuple(marked_simplex)
        assert len(self._marked_simplex) == 1, (
            "Orbits for simplexes that are not nodes are currently not supported."
            "Though it should be possible to implement them."
        )

    @property
    def sc(self) -> SimplicialComplexWrapper:
        return self._sc
    
    @property
    def marked_simplex(self) -> tuple[int, ...]:
        return self._marked_simplex

    def __eq__(self, other) -> bool:
        if self._marked_simplex != other._marked_simplex:
            return False
        if not equal(self._sc, other._sc):
            return False
        return True

    def __hash__(self):
        return hash((tuple(self._sc.f_vector.tolist()), self._marked_simplex))


@dataclass
class OrbitEntry:
    entry: tuple[int, ...]
    orbit: Orbit


##################### Utils #####################

def sc_to_simplices(
        sc: SimplicialComplex,
        compact: bool = False
) -> tuple[tuple[int, ...], ...]:
    simplices = tuple(map(
        lambda x: tuple(sorted(x.tolist())),
        itertools.chain(*sc.simplices)
    ))

    if compact:
        simplices_compact: set[tuple[int, ...]] = set()

        for simplex in simplices:
            is_face = False
            for other in simplices_compact.copy():
                if set(simplex) <= set(other):
                    is_face = True
                    break
                elif set(other) <= set(simplex):
                    simplices_compact.remove(other)
            if not is_face:
                simplices_compact.add(simplex)

        simplices = tuple(simplices_compact)

    return simplices


def construct_sc_from_simplices(
        simplices_list: tp.Sequence[tp.Sequence[int]],
        check_nodes: bool = True,
) -> SimplicialComplexWrapper:
    """
    Creates SimplicialComplex from the list of simplices.
    Simplices in the list can be ordered in any way.
    """
    simplices: list[list[list[int]]] = []
    for simplex in simplices_list:
        dim = len(simplex) - 1
        while len(simplices) <= dim:
            simplices.append([])
        simplices[dim].append(list(simplex))

    simplices_extended: list[np.ndarray] = [np.empty([]) for _ in range(len(simplices))]
    for dim in reversed(range(len(simplices))):
        simplices_extended[dim] = np.sort(np.unique(np.array(simplices[dim]), axis=0))
        if dim > 0:
            for u in range(dim+1):
                mask = np.arange(dim+1) != u
                simplices[dim-1].extend(simplices_extended[dim][:, mask].tolist())

    sc = SimplicialComplex()
    sc.simplices = simplices_extended  # type: ignore

    return SimplicialComplexWrapper(sc, check_nodes)


def is_connected(sc: SimplicialComplexWrapper) -> bool:
    if len(sc.simplices) < 2:
        return False
    return nx.is_connected(sc.line_graph(0, 1))

def get_canonical_simplices(simplices: np.ndarray) -> np.ndarray:
    simplices = np.sort(simplices, axis=1)  # Sort nodes in each simplex
    indices = np.lexsort(simplices.T)  # Sort simplices
    return simplices[indices]


def equal(first: SimplicialComplexWrapper, second: SimplicialComplexWrapper) -> bool:
    if first.f_vector.tolist() != second.f_vector.tolist():
        return False
    for first_arr, second_arr in zip(first.simplices, second.simplices):
        if first_arr.shape != second_arr.shape:
            return False
        if not np.all(
                get_canonical_simplices(first_arr)
                ==
                get_canonical_simplices(second_arr)
        ):
            return False
    return True


def permute_orbit(orbit: Orbit, permutation: tp.Sequence[int]) -> Orbit:
    permutation_array = np.array(permutation)

    permuted_simplices: list[list[int]] = []
    for simplex in itertools.chain(*orbit.sc.simplices):
        permuted_simplex = permutation_array.take(simplex)
        permuted_simplices.append(permuted_simplex)
    permuted_sc = construct_sc_from_simplices(permuted_simplices)

    permuted_marked_simplex = tuple(
        permutation_array.take(orbit.marked_simplex)
    )
    
    return Orbit(permuted_sc, permuted_marked_simplex)


def isomorphic(first: Orbit, second: Orbit) -> bool:
    nodes = first.sc.simplices[0].reshape(-1).tolist()
    if first.sc.f_vector.tolist() != second.sc.f_vector.tolist():
        return False
    for permutation in itertools.permutations(nodes):
        if permute_orbit(first, permutation) == second:
            return True
    return False


def sort_orbits(orbits: tp.Sequence[Orbit]) -> list[Orbit]:
    key = lambda orbit: -len(list(itertools.chain(*orbit.sc.simplices)))
    return sorted(orbits, key=key)


def is_antichain(simplices: tp.Sequence[tp.Sequence[int]]) -> bool:
    simplices_sets = [set(simplex) for simplex in simplices]
    for i in range(len(simplices)):
        for j in range(i+1, len(simplices)):
            if simplices_sets[i] <= simplices_sets[j]:
                return False
            if simplices_sets[j] <= simplices_sets[i]:
                return False
    return True


@functools.cache
def get_all_simplet_orbits(
        num_nodes: int,
        marked_simplex: tuple[int] = (0,),
        use_cached: bool = True,
) -> list[Orbit]:
    """
    Returns all possible orbits of the given size.
    Orbits are returned in non-increasing order. This means that
    if the orbit A appears before the orbit B, than A can't be suborbit of B.
    """
    if use_cached and (3 <= num_nodes <= 4):
        if num_nodes == 3:
            simplices_list = CACHED_ORBITS_3
        elif num_nodes == 4:
            simplices_list = CACHED_ORBITS_4
        else:
            raise ValueError(f"Orbit size {num_nodes} is not cached")
        orbits = [
            Orbit(construct_sc_from_simplices(simplices), marked_simplex)
            for simplices in simplices_list
        ]
    else:
        orbits = []

        nodes = list(range(num_nodes))

        complete_simplex_list = list(itertools.chain.from_iterable(
            itertools.combinations(nodes, dim+1)
            for dim in range(1, len(nodes))
        ))

        for simplex_list in itertools.chain.from_iterable(
                itertools.combinations(complete_simplex_list, num_simplices)
                for num_simplices in range(len(complete_simplex_list) + 1)
        ):
            if len(simplex_list) == 0:
                continue

            simplices: list[list[int]] = []
            simplices.extend([list(simplex) for simplex in simplex_list])

            if not is_antichain(simplices):
                continue

            simplices.extend([[node] for node in nodes])
            sc = construct_sc_from_simplices(simplices)
            orbit = Orbit(sc, marked_simplex)

            if not is_connected(sc):
                continue

            is_new_orbit = True
            for other_orbit in orbits:
                assert isomorphic(orbit, orbit)
                if isomorphic(orbit, other_orbit):
                    is_new_orbit = False
                    break

            if is_new_orbit:
                orbits.append(orbit)

    orbits = sort_orbits(orbits)
    return orbits


def construct_induced_subcomplex(
        sc: SimplicialComplexWrapper,
        nodes: tp.Sequence[int]
) -> SimplicialComplexWrapper:
    subcomplex_simplices: list[list[int]] = []

    for simplex in itertools.chain.from_iterable(
        itertools.combinations(nodes, dim)
        for dim in range(1, len(nodes)+1)
    ):
        simplex = tuple(sorted(simplex))
        if simplex in sc.simplices_set():
            subcomplex_simplices.append([nodes.index(u) for u in simplex])

    subcomplex = construct_sc_from_simplices(subcomplex_simplices)
    return subcomplex


def iterate_connected_entries(
        sc: SimplicialComplexWrapper,
        entry_size: int,
        current_subset: list[int] = [],
        marked_simplex: tuple[int, ...] = (0,)
) -> tp.Iterable[list[int]]:
    """
    Node at index 0 is marked, while all other nodes are unordered
    """
    assert marked_simplex == (0, ), \
        "Other marked simplices are not supported yet"

    if len(current_subset) == entry_size:
        for permutation in itertools.permutations(current_subset[1:]):
            yield [current_subset[0], *permutation]
    else:
        if len(current_subset) == 0:
            candidates: set[int] = set(range(sc.simplices[0].shape[0]))
        else:
            candidates = set()
            for node in current_subset:
                candidates |= sc.adj(node)

        for node in candidates:
            if node in current_subset:
                continue
            current_subset.append(node)
            yield from iterate_connected_entries(
                sc, entry_size, current_subset
            )
            current_subset.pop()


def iterate_orbit_entries(
        sc: SimplicialComplexWrapper,
        orbits: tp.Sequence[Orbit],
) -> tp.Iterable[OrbitEntry]:
    nodes = list(range(sc.simplices[0].shape[0]))
    marked_simplex = orbits[0].marked_simplex
    orbits_set = set(orbits)
    orbit_size = orbits[0].sc.simplices[0].shape[0]
    for orbit in orbits:
        assert orbit.sc.simplices[0].shape[0] == orbit_size

    for entry_subset in itertools.product(nodes, repeat=1):
        visited: set[tuple[int, ...]] = set()
        for entry in map(
                tuple,
                iterate_connected_entries(sc, orbit_size, list(entry_subset))
        ):
            entry_unordered = (entry[0], *sorted(entry[1:]))
            if entry_unordered in visited:
                continue
            subcomplex = construct_induced_subcomplex(sc, entry)
            orbit = Orbit(subcomplex, marked_simplex)
            if orbit in orbits_set:
                yield OrbitEntry(entry, orbit)
                visited.add(entry_unordered)

##################### Bruteforce #####################

def compute_orbit_counts_bruteforce(
        sc: SimplicialComplex,
        orbit_size: int,
) -> dict[Orbit, list[int]]:
    wrapped_sc = SimplicialComplexWrapper(sc)
    orbits = get_all_simplet_orbits(orbit_size)
    counts: dict[Orbit, list[int]] = {
        orbit: [0 for _ in range(sc.simplices[0].shape[0])]
        for orbit in orbits
    }
    for orbit_entry in iterate_orbit_entries(wrapped_sc, orbits):
        counts[orbit_entry.orbit][orbit_entry.entry[0]] += 1
    return counts

##################### ORCA #####################

def construct_suborbit(orbit: Orbit) -> Orbit:
    nodes = list(range(orbit.sc.simplices[0].shape[0]))
    suborbit: Orbit | None = None
    best_degree: int = -1
    for node in nodes:
        if (node,) == orbit.marked_simplex:
            continue
        remaining_nodes = nodes.copy()
        remaining_nodes.remove(node)
        new_sc = construct_induced_subcomplex(orbit.sc, remaining_nodes)
        if is_connected(new_sc):
            degree = len(orbit.sc.adj(node))
            if (suborbit is None) or (degree < best_degree):
                suborbit = Orbit(new_sc, orbit.marked_simplex)
                best_degree = degree
    assert suborbit is not None, "There should be new orbit!"
    return suborbit


@functools.cache
def construct_extensions(
        suborbit: Orbit,
        orbit: Orbit
) -> tp.Sequence[Extension]:
    extensions: set[Extension] = set()
    nodes = list(range(suborbit.sc.simplices[0].shape[0]))
    node_subsets = list(itertools.chain.from_iterable(
        itertools.combinations(nodes, subset_size)
        for subset_size in range(1, len(nodes)+1)
    ))
    new_node = len(nodes)
    for extension in itertools.chain.from_iterable(
            itertools.combinations(node_subsets, extension_size)
            for extension_size in range(1, len(node_subsets) + 1)
    ):
        extended_simplices: list[list[int]] = []
        extended_simplices.append([new_node])
        for simplex in itertools.chain(*suborbit.sc.simplices):
            extended_simplices.append(simplex)
        for subset in extension:
            extended_simplices.append([new_node] + list(subset))
        extended_sc = construct_sc_from_simplices(extended_simplices)
        extended_orbit = Orbit(extended_sc, suborbit.marked_simplex)
        if isomorphic(extended_orbit, orbit):
            # The line below de-duplicates extensions
            extension = sc_to_simplices(
                construct_sc_from_simplices(extension, check_nodes=False),
            )
            extensions.add(extension)
    return list(extensions)


def compute_rhs_dict(
        sc: SimplicialComplexWrapper,
        orbits: tp.Sequence[Orbit],
        suborbits: tp.Sequence[Orbit],
        extensions: tp.Sequence[tp.Sequence[Extension]],
) -> dict[Orbit, list[int]]:
    num_nodes = sc.simplices[0].shape[0]
    rhs = {orbit: [0 for _ in range(num_nodes)] for orbit in orbits}
    for suborbit_entry in iterate_orbit_entries(sc, suborbits):
        for i, suborbit in enumerate(suborbits):
            if suborbit_entry.orbit == suborbit:
                for extension in extensions[i]:
                    entry = suborbit_entry.entry
                    extension_in_complex = tuple([
                        tuple([entry[u] for u in subset])
                        for subset in extension
                    ])
                    rhs[orbits[i]][entry[suborbit.marked_simplex[0]]] += (
                        sc.compute_num_common_neighbors(extension_in_complex)
                        -
                        suborbit.sc.compute_num_common_neighbors(extension)
                    )
    return rhs


def compute_lhs_coef_for_orbit(
        orbit: Orbit,
        suborbit: Orbit,
        extensions: tp.Sequence[Extension],
) -> int:
    coef = 0
    for excluded_node in range(orbit.sc.simplices[0].shape[0]):
        if excluded_node == orbit.marked_simplex[0]:
            continue
        remaining_nodes_unordered = list(range(orbit.sc.simplices[0].shape[0]))
        remaining_nodes_unordered.remove(excluded_node)
        for remaining_nodes in itertools.permutations(remaining_nodes_unordered):
            if remaining_nodes[0] != 0:
                continue
            remaining_complex = construct_induced_subcomplex(
                orbit.sc, remaining_nodes)
            remaining_orbit = Orbit(remaining_complex, suborbit.marked_simplex)
            if remaining_orbit == suborbit:
                excluded_node_neighbors: set[tuple[int, ...]] = set()
                for simplex in itertools.chain(*orbit.sc.simplices):
                    simplex = simplex.tolist()
                    if excluded_node not in simplex:
                        continue
                    simplex.remove(excluded_node)
                    simplex = tuple([remaining_nodes.index(u) for u in simplex])
                    excluded_node_neighbors.add(simplex)
                for extension in extensions:
                    if set(extension) <= excluded_node_neighbors:
                        coef += 1
                break
    return coef


def get_equation_lhs(
        sc: SimplicialComplexWrapper,
        suborbit: Orbit,
        extensions: tp.Sequence[Extension],
        orbit_counts_dict: dict[Orbit, list[int]],
) -> list[int]:
    lhs_counts: list[int] = [0 for _ in range(sc.simplices[0].shape[0])]
    # NOTE: it actually includes orbit for which equation is being constructed
    # But this is not a problem cause orbit_counts for it is filled with zeros
    for orbit, orbit_counts in orbit_counts_dict.items():
        coef = compute_lhs_coef_for_orbit(orbit, suborbit, extensions)
        for u in range(len(lhs_counts)):
            lhs_counts[u] += coef * orbit_counts[u]
    return lhs_counts


def solve_equation(
        sc: SimplicialComplexWrapper,
        orbit: Orbit,
        suborbit: Orbit,
        extensions: tp.Sequence[Extension],
        orbit_counts_dict: dict[Orbit, list[int]],
        rhs_dict: dict[Orbit, list[int]],
) -> list[int]:
    # suborbit = construct_suborbit(orbit)  # TODO: get them from outside
    # extensions = construct_extensions(suborbit, orbit)
    rhs = rhs_dict[orbit]
    lhs = get_equation_lhs(sc, suborbit, extensions, orbit_counts_dict)
    counts = [r-l for l, r in zip(lhs, rhs)]
    coef = compute_lhs_coef_for_orbit(orbit, suborbit, extensions)
    for u in range(len(counts)):
        assert counts[u] % coef == 0, f"{counts[u], coef}"
        counts[u] //= coef
    return counts


def compute_complete_orbit_count(
        sc: SimplicialComplexWrapper,
        orbit_size: int
) -> list[int]:
    counts = [0 for _ in range(sc.simplices[0].shape[0])]

    dim = orbit_size - 1
    if len(sc.simplices) <= dim:
        return counts
    
    for simplex in sc.simplices[dim]:
        for u in simplex.tolist():
            counts[u] += 1

    return counts


def orca(
        sc: SimplicialComplex,
        orbit_size: int,
) -> dict[Orbit, list[int]]:
    wrapped_sc = SimplicialComplexWrapper(sc)
    orbits = get_all_simplet_orbits(orbit_size)
    orbit_counts_dict: dict[Orbit, list[int]] = dict()

    suborbits = [construct_suborbit(orbit) for orbit in orbits]
    extensions = [
        construct_extensions(suborbit, orbit)
        for suborbit, orbit in zip(suborbits, orbits)
    ]

    # NOTE: rhs requires iterating over the suborbit entries,
    # so it is more efficient to precompute them via single iteration 
    # rather than have separate iteration over all entries for each suborbit
    rhs_dict = compute_rhs_dict(wrapped_sc, orbits, suborbits, extensions)

    orbit_counts_dict[orbits[0]] = compute_complete_orbit_count(
        wrapped_sc, orbit_size)

    for i in range(1, len(orbits)):
        orbit_counts_dict[orbits[i]] = solve_equation(
            sc=wrapped_sc,
            orbit=orbits[i],
            suborbit=suborbits[i],
            extensions=extensions[i],
            orbit_counts_dict=orbit_counts_dict,
            rhs_dict=rhs_dict,
        )
    return orbit_counts_dict
