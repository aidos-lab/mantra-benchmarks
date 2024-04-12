import argparse
import json
import re


def process_triangulation_line(line):
    """'
    Parses a single line of the triangluation and returns the fields in a
    dictionary. The input string has to have the format
    'manifold_{dim}_{n_vert}_{non-unique-id}=[[1,2,3],...,[1,4,3]]'
    Returns dictionary with triangulation as list of lists, dimension and number
    of vertices.
    """
    line = re.split("=", line)

    # Parse dimension and n_vertices from id.
    match = re.match("manifold_(\d+)_(\d+)", line[0])

    # json.loads loads string into List[List[int]].
    return {
        line[0]: {
            "id": line[0],
            "dimension": match.group(1),
            "n_vertices": match.group(2),
            "triangulation": json.loads(line[1]),
        }
    }


def process_triangulation(content: str):
    lines = content.removesuffix("\n\n")
    lines = re.split("\n\n", lines)
    lines = [re.sub("\n\s+", "", line) for line in lines]

    dicts = [process_triangulation_line(line) for line in lines]
    return {k: v for d in dicts for k, v in d.items()}


def process_type_line(line: str) -> dict:
    """
    Parses a single line with the homology type. The line has to have the format
    'manifold_{dim}_{n_vert}_{non-unique-id}:  ( {orientable} ; {genus} ) =
    {name}'. The name field is optional and may be omitted. The function returns
    a dictionary with all fields parsed.

    Examples:
    - 'manifold_2_9_645:  ( - ; 1 ) = RP^2'
    - 'manifold_2_9_646:  ( - ; 2 ) = Klein bottle'
    - 'manifold_2_9_587:  ( - ; 3 )'

    Input:
        String
    Output:
        Dict
    """
    match = re.match(
        r"(manifold_.*):\s+\( ([+-])\s;\s(\d)\s\)(\s=\s)?(.*)?", line
    )
    return {
        match.group(1): {
            "id": match.group(1),
            "orientable": True if match.group(2) == "+" else False,
            "genus": match.group(3),
            "name": match.group(5),
        }
    }


def process_type(content):
    lines = content.removesuffix("\n").split("\n")
    dicts = [process_type_line(line) for line in lines]
    return {k: v for d in dicts for k, v in d.items()}


def process_homology_line(line):
    match = re.match(r"(manifold_.*):\s+\((.*)\)", line)
    tc, bn = [], []
    for rank in match.group(2).split(", "):
        rank_match = re.match("(\d+)(\s\+\s)?(.*)?", rank)
        bn.append(int(rank_match.group(1)))
        tc.append(rank_match.group(3))
    return {match.group(1): {"torsion_coefficients": tc, "betti_numbers": bn}}


def process_homology(lines):
    lines = lines.removesuffix("\n").split("\n")
    dicts = [process_homology_line(line) for line in lines]
    return {k: v for d in dicts for k, v in d.items()}


def process_manifolds(
    filename_triangulation, filename_homology=None, filename_type=None
):
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

    # Merge all together
    for manifold in triangulations:
        triangulations[manifold].update(
            homology_groups[manifold] | types[manifold]
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
    if args.output:
        with open(args.output, "w") as f:
            f.write(result)
    else:
        print(result)
