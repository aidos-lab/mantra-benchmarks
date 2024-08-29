from toponetx import SimplicialComplex


def barycentric_subdivision(K: SimplicialComplex) -> SimplicialComplex:
    # Create a new SimplicialComplex to store the subdivision
    Sd_K = SimplicialComplex()

    new_simplices = {dim: set() for dim in range(K.dim + 1)}

    # Add new vertices to Sd_K. Each simplex of Sd_K is a chain of simplices of K
    for simplex in K.simplices:
        new_simplices[0].add((simplex,))

    # Give now an index to each simplex
    simplex_to_index = {simplex[0]: i for i, simplex in enumerate(new_simplices[0])}

    # Now, we add simplices from dimension 1 to K.dim
    for dim in range(1, K.dim + 1):
        # Get all simplices of the previous dimension, and try to add more simplices to the chain
        previous_simplices = new_simplices[dim - 1]
        for simplex_sub in previous_simplices:
            last_simplex = simplex_sub[-1]
            for simplex in K.simplices:
                # Check if simplex is a face of simplex_sub
                if last_simplex < simplex:
                    new_simplices[dim].add(simplex_sub + (simplex,))
    # Now convert the simplices to indexes
    all_simplices = []
    for dim in range(K.dim + 1):
        for simplex in new_simplices[dim]:
            all_simplices.append([simplex_to_index[or_simplex] for or_simplex in simplex])
    # Add the simplices to the new SimplicialComplex
    print()
    Sd_K.add_simplices_from(all_simplices)
    return Sd_K, simplex_to_index


def recursive_barycentric_subdivision(K: SimplicialComplex, number_of_transformations: int) -> SimplicialComplex:
    Sd_K = K
    for _ in range(number_of_transformations):
        Sd_K, _ = barycentric_subdivision(Sd_K)
    return Sd_K
