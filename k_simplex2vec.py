import gudhi
import numpy as np
from numpy import matlib
import random
import gensim
from gensim.models import Word2Vec
import scipy.sparse

# Some function that will be useful for the rest of the code


def signed_faces(s):
    ## returns the faces of the simplex s
    ret = []
    for i in range(0, len(s)):
        ret.append(((-1) ** i, s[0:i] + s[i + 1 : len(s)]))
    return ret


def signed_cofaces(s, cplx):
    # returns all cofaces (of codim 1)  of simplex s
    return [(sign(s, x), x) for (x, eps) in cplx.get_cofaces(s, 1)]


def sign(s, t):  # Sign of s in t.
    if len(t) != len(s) + 1:
        return None

    for i in range(0, len(s)):
        if s[i] != t[i]:
            return (-1) ** i

    return (-1) ** len(s)


def hacky_get_idx(s, cplx):
    # Get index of the simplices
    i = cplx.filtration(s)
    assert i.is_integer()
    return int(i)


def assemble(cplx, k, scheme="uniform", laziness=None):
    ## Assmeble the transition matrix
    # We are using this incredibly ugly hack to store the indices of the simplices
    # as filtration values since keys are not accessible
    # through the GUDHI Python API.
    assert cplx.num_simplices() < 16777217

    assert scheme in ["uniform", "uniform-lazy", "uniform-multicount"]
    if scheme == "uniform-lazy":
        assert laziness is not None
        assert laziness >= 0 and laziness <= 1

    simplices = [s for (s, eps) in cplx.get_filtration()]

    ordering = []
    N = 0
    for s in simplices:
        if len(s) == k + 1:
            cplx.assign_filtration(s, float(N))
            ordering.append(s)
            N += 1
        else:
            cplx.assign_filtration(s, np.inf)

    cplx.initialize_filtration()

    row_inds = []
    col_inds = []
    data = []

    for s, i in cplx.get_filtration():
        if i >= N:
            break

        assert i.is_integer()
        i = int(i)

        # uniform, uniform-lazy, uniform-multicount
        if scheme.startswith("uniform"):

            s_faces = signed_faces(s)
            s_cofaces = signed_cofaces(s, cplx)

            s_up = []
            for a, t in s_cofaces:
                s_up += [(-a * b, u) for (b, u) in signed_faces(t)]

            s_down = []
            for a, t in s_faces:
                s_down += [(-a * b, u) for (b, u) in signed_cofaces(t, cplx)]

            ## We are not considering orientations so we set all signs to 1
            s_up = [(1, t) for (foo, t) in s_up]
            s_down = [(1, t) for (foo, t) in s_down]

            if scheme == "uniform-multicount":
                s_neigh_idxs = [(a, hacky_get_idx(t, cplx)) for (a, t) in s_down + s_up]
            else:
                s_neigh_idxs = list(
                    set([(a, hacky_get_idx(t, cplx)) for (a, t) in s_down + s_up])
                )

            if scheme == "uniform-lazy":
                if len(s_neigh_idxs) == 1:
                    probs = 0.0
                else:
                    num_self_neigh = 0
                    for sgn, j in s_neigh_idxs:
                        if j == i:
                            num_self_neigh += 1
                    probs = (1.0 - laziness) / (len(s_neigh_idxs) - num_self_neigh)
            else:
                probs = 1.0 / len(s_neigh_idxs)

            for sgn, j in s_neigh_idxs:
                row_inds.append(i)
                col_inds.append(j)
                if scheme == "uniform-lazy" and j == i:
                    data.append(laziness)
                else:
                    data.append(probs)

    return scipy.sparse.csr_matrix((data, (row_inds, col_inds)), shape=(N, N))


def walk(smplx, walk_length, P):
    ## Performs a single random walk of fixed length starting at smplx
    # smplx = starting simplex of the random walk
    # P = precomputed transition matrix on the complex containing smplx
    # walk_length = length of the random walk
    c = np.arange(P.shape[0])
    RW = []
    RW.append(smplx)
    for i in range(walk_length):
        smplx = np.random.choice(c, size=1, p=P[smplx])[0]
        RW.append(smplx)
    return RW


def RandomWalks(walk_length, number_walks, P, seed=None):
    ## Performs a fixed number of random walks at each $k$-simplex
    Walks = []  ## List where we store all random walks of length walk_length
    for i in range(number_walks):
        for smplx in range(P.shape[0]):
            Walks.append(walk(smplx, walk_length, P))
    if seed != None:
        np.random.seed(seed)
        np.random.shuffle(Walks)
    else:
        np.random.shuffle(Walks)
    return Walks


def save_random_walks(Walks, filename):
    ## Writes the walks in a .txt file
    file = open(filename, "a")
    for walk in Walks:
        L = str(walk)[1:-1] + "\n"
        file.write(L)
    file.close()


def load_walks(filename):
    ## Loads a file with precomputed random walks
    file = open(filename, "r")
    lines = file.readlines()
    walks = list()
    for line in lines:
        walk = list()
        line = line[0:-1]
        newline = line.split("], [")
        for el in newline:
            step = [int(s) for s in el.split(", ")]
            walk.append(step)
        walks.append(walk[0])
    return walks


def Embedding(Walks, emb_dim, epochs=5, filename="k-simplex2vec_embedding.model"):
    ## Performs the embedding of the $k$-simplices using the gensim word2vec package
    walks_str = []
    for i in range(len(Walks)):
        ls_temp = []
        for j in range(len(Walks[i])):
            string = str(Walks[i][j]).replace(" ", "")
            ls_temp.append(string)
        walks_str.append(ls_temp)

    model = Word2Vec(
        walks_str,
        vector_size=emb_dim,
        window=3,
        min_count=0,
        sg=1,
        workers=1,
        epochs=epochs,
    )
    model.save(filename)
    return model
