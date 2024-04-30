"""Convert from lexicographical format to line-based format.

The purpose of this script is to convert triangulations from
a lexicographical format to a line-based format. In the old,
i.e. lexicographical, format, triangulations can span *more*
than one line, and every line contains a different number of
vertices. This makes parsing the format cumbersome.

As an alternative, this script parses the triangulations and
represents them as JSON objects. This facilitates *storing*,
as well as *processing* the data types.
"""

import argparse
import json
import re
import pydantic
from typing import List, Optional, Dict
import pandas as pd
import numpy as np


class Triangulation(pydantic.BaseModel):
    id: str
    triangulation: List[List[int]]
    dimension: int
    n_vertices: int

    @pydantic.model_validator(mode="after")
    def check_model(self):
        # Check if triangulations have the same length of 3.
        if not all([len(item) == 3 for item in self.triangulation]):
            raise pydantic.ValidationError(
                "Every element in the list should have length 3."
            )

        # Triangulation has 1 based indexing, so max is number of vertices.
        if not np.array(self.triangulation).max() == self.n_vertices:
            raise ValueError(
                f"Number of vertices in the triangulation ({np.array(self.triangulation).max()}) does not coincide with n_vertices ({self.n_vertices})"
            )

        # Check if dimension is correct.
        assert self.dimension == len(self.triangulation[0]) - 1


class TopologicalType(pydantic.BaseModel):
    id: str
    orientable: bool
    genus: int
    name: Optional[str]


class Homology(pydantic.BaseModel):
    id: str
    torsion_coefficients: List[str]
    betti_numbers: List[int]

    @pydantic.model_validator(mode="after")
    def check_model(self):
        assert len(self.betti_numbers) == 3
        assert len(self.torsion_coefficients) == 3


def process_triangulation_line(line: str) -> Triangulation:
    """Parses a single line of the triangluation.

    A triangulation is represented (following the original data format)
    as a newline-separated string of vertex indices. These indices will
    be parsed into a (nested) array of integers and returned.

    Example:
    'manifold_{dim}_{n_vert}_{non-unique-id}=[[1,2,3],...,[1,4,3]]'

    Returns
    -------
    dict
        Dictionary, with keys indicating the respective triangulation
        and values being strings corresponding to homology groups, or
        type information, respectively. No further processing of each
        string is attempted.
    """
    line = re.split("=", line)

    # Parse dimension and n_vertices from id.
    match = re.match("manifold_(\d+)_(\d+)", line[0])

    # json.loads loads string into List[List[int]].
    return Triangulation(
        id=line[0],
        dimension=match.group(1),
        n_vertices=match.group(2),
        triangulation=json.loads(line[1]),
    )


def process_triangulation(content: str) -> List[Triangulation]:
    lines = content.removesuffix("\n\n")
    lines = re.split("\n\n", lines)
    lines = [re.sub("\n\s+", "", line) for line in lines]

    return [process_triangulation_line(line) for line in lines]


def process_type_line(line: str) -> TopologicalType:
    """Parses a single line with the homology type.

    The line has to have the format
    'manifold_{dim}_{n_vert}_{non-unique-id}:  ( {orientable} ; {genus} ) =
    {name}'. The name field is optional and may be omitted. The function returns
    a dictionary with all fields parsed.

    Examples:
    - 'manifold_2_9_645:  ( - ; 1 ) = RP^2'
    - 'manifold_2_9_646:  ( - ; 2 ) = Klein bottle'
    - 'manifold_2_9_587:  ( - ; 3 )'

    Parameters
    ----------
    line : string
        A single line to parse in the format above.

    Returns:
        Dictionary containing the manifold id, the corresponding homology groups,
        orientation and if present the name.
    """
    match = re.match(
        r"(manifold_.*):\s+\( ([+-])\s;\s(\d)\s\)(\s=\s)?(.*)?", line
    )
    return TopologicalType(
        id=match.group(1),
        orientable=True if match.group(2) == "+" else False,
        genus=int(match.group(3)),
        name=match.group(5),
    )


def process_type(content: str) -> List[TopologicalType]:
    lines = content.removesuffix("\n").split("\n")
    return [process_type_line(line) for line in lines]


def process_homology_line(line: str) -> Homology:
    match = re.match(r"(manifold_.*):\s+\((.*)\)", line)
    tc, bn = [], []
    for rank in match.group(2).split(", "):
        rank_match = re.match("(\d+)(\s\+\s)?(.*)?", rank)
        bn.append(int(rank_match.group(1)))
        tc.append(rank_match.group(3))

    return Homology(
        id=match.group(1), torsion_coefficients=tc, betti_numbers=bn
    )


def process_homology(lines: str) -> List[Homology]:
    lines = lines.removesuffix("\n").split("\n")
    return [process_homology_line(line) for line in lines]


def merge_triangulation(
    triangulation: List[Triangulation],
    homology_groups: List[Homology],
    types: List[TopologicalType],
) -> List[Triangulation]:
    df_triangulation = pd.DataFrame.from_records(
        [tr.__dict__ for tr in triangulation]
    ).set_index("id")

    if homology_groups:
        df_homology = pd.DataFrame.from_records(
            [h.__dict__ for h in homology_groups]
        ).set_index("id")
        df_triangulation = df_triangulation.join(df_homology, "id")

    if types:
        df_types = pd.DataFrame.from_records(
            [t.__dict__ for t in types]
        ).set_index("id")
        df_triangulation = df_triangulation.join(df_types, "id")
    return df_triangulation.to_dict(orient="records")


def process_manifolds(
    filename_triangulation: str,
    filename_homology: str | None = None,
    filename_type: str | None = None,
) -> List[Dict]:
    homology_groups, types = {}, {}

    # Parse triangulations
    with open(filename_triangulation) as f:
        lines = f.read()
    triangulations = process_triangulation(lines)

    # Parse homology
    if filename_homology:
        with open(filename_homology) as f:
            lines = f.read()
        homology_groups = process_homology(lines)

    # Parse type
    if filename_type:
        with open(filename_type) as f:
            lines = f.read()
        types = process_type(lines)

    if filename_homology or filename_type:
        triangulations = merge_triangulation(
            triangulations, homology_groups, types
        )

    return triangulations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "triangulation", type=str, help="Input triangulation file"
    )
    parser.add_argument(
        "-H",
        "--homology",
        type=str,
        help="Homology information for triangulations (optional)",
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        help="Type information for triangulations (optional)",
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Output file (optional)"
    )

    args = parser.parse_args()

    triangulations = process_manifolds(
        args.triangulation, args.homology, args.type
    )

    result = json.dumps(triangulations, indent=2)
    # if args.output:
    #     with open(args.output, "w") as f:
    #         f.write(result)
    # else:
    #     print(result)


def process_train_test_split_orientability(
    filename_train_test_split_orientability: str,
):
    with open(filename_train_test_split_orientability, "r") as f:
        indices_raw = f.readlines()
    for indices_line in indices_raw:
        type_indices, indices = indices_line.split(":")
        if type_indices == "Train indices":
            train_indices = [int(idx) for idx in indices.strip().split(" ")]
        elif type_indices == "Test indices":
            test_indices = [int(idx) for idx in indices.strip().split(" ")]
        else:
            raise NotImplementedError("Unknown type of indices")
    return np.array(train_indices), np.array(test_indices)
