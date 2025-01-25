import numpy as np


def pad_atoms_if_needed(atoms, length_pos_enc, padding):
    if not padding and atoms.shape[1] < length_pos_enc:
        raise ValueError(
            f"If padding is not used, the number or atoms in the dictionary of topological slepians"
            f"must be greater or equal to the length of the positional encoding. ç"
            f"Got {length_pos_enc} and {atoms.shape[1]}."
        )
    if length_pos_enc > atoms.shape[1]:
        atoms = np.pad(
            atoms,
            ((0, 0), (0, length_pos_enc - atoms.shape[1])),
            "constant",
            constant_values=(0, 0),
        )
    return atoms[:, :length_pos_enc]
