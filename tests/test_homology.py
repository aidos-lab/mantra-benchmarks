"""Test correctness of homology information in an input file.

This test takes a JSON file containing a set of triangulations and
checks their Betti numbers. To this end, a simplex tree is created
from the triangulation and its homology is calculated.
"""

import gudhi as gd

import argparse
import itertools
import json


def build_simplex_tree(top_level_simplices):
    simplices = set([tuple(s) for s in top_level_simplices])
    max_dim = len(next(iter(simplices)))

    for simplex in top_level_simplices:
        for dim in range(1, max_dim):
            simplices.update(s for s in itertools.combinations(simplex, r=dim))

    # Get every complex into lexicographic order. This
    # requires converting everything back to a list.
    simplices = list(simplices)
    simplices.sort()
    simplices.sort(key=len)

    st = gd.SimplexTree()
    for simplex in simplices:
        st.insert(simplex)

    return st


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT", type=str, help="Input file")

    args = parser.parse_args()

    with open(args.INPUT) as f:
        data = json.load(f)

    for data_ in data:
        simplex_tree = build_simplex_tree(data_["triangulation"])
        persistence_pairs = simplex_tree.persistence(persistence_dim_max=True)

        dimensions = range(data_["dimension"] + 1)

        betti_numbers = []

        for dimension in dimensions:
            pairs = [
                (a, b) for dim, (a, b) in persistence_pairs if dim == dimension
            ]

            betti_numbers.append(len(pairs))

        assert data_["betti_numbers"] == betti_numbers
