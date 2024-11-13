from dataclasses import dataclass


@dataclass
class Interaction:
    """
    Data class to represent an interaction between two dimensions in a tensor diagram. The interaction consists of two
    nodes and an edge between them, representing the flow of information from one dimension to the other.
    """

    in_dim: int
    out_dim: int

    def __str__(self):
        return f"{self.in_dim}->{self.out_dim}"

    @staticmethod
    def from_string(interaction_str: str) -> "Interaction":
        """
        Create an Interaction object from a string representation of the interaction.
        :param interaction_str: The string representation of the interaction.
        :return: The Interaction object.
        """
        # Check if the pair contains exactly one "->"
        if interaction_str.count("->") != 1:
            raise ValueError(
                "Invalid pair format. Each pair should contain exactly one '->'."
            )
        in_dim, out_dim = interaction_str.split("->")
        # Check if both N and M are valid integers
        if not in_dim.isdigit() or not out_dim.isdigit():
            raise ValueError("Invalid pair format. Both N and M should be integers.")
        return Interaction(int(in_dim), int(out_dim))

    def __hash__(self):
        return hash((self.in_dim, self.out_dim))


class TensorDiagram:
    """
    Class to represent a tensor diagram for only one layer of a neural network, this is, a tensor diagram with two
    levels, the input level, and the output level. The tensor diagram is represented as a string, where each pair of
    dimensions is separated by a comma, and the input and output dimensions are separated by an arrow "->". For example,
    the string "1->2,2->3,3->3" represents a tensor diagram with three interactions, where the first interaction is
    between dimensions 1 and 2, the second interaction is between dimensions 2 and 3, and the third interaction is
    between dimensions 3 and 3. Note that the dimensions must be non-negative integers.
    """

    def __init__(self, tensor_diagram_string_representation: str):
        self.tensor_diagram_string_representation = tensor_diagram_string_representation
        self._interactions = None
        self._considered_dimensions = None
        self.process()

    def process(self) -> None:
        """
        Process the input string and store the interactions in a list, if the string is valid.
        """
        _interactions = []
        _considered_dimensions = set()
        # Remove any whitespace and split the string by commas
        pairs = self.tensor_diagram_string_representation.replace(" ", "").split(",")
        for pair in pairs:
            interaction = Interaction.from_string(pair)
            _interactions.append(interaction)
            # Add the dimensions to the set of considered dimensions
            _considered_dimensions.add(interaction.in_dim)
            _considered_dimensions.add(interaction.out_dim)
        # Set the interactions and considered dimensions as immutable
        self._interactions = frozenset(_interactions)
        self._considered_dimensions = frozenset(_considered_dimensions)

    @property
    def considered_dimensions(self) -> set[int]:
        return self._considered_dimensions

    @property
    def interactions(self) -> set[Interaction]:
        return self._interactions
