from enum import Enum


def enum_from_str_id(id: str, enum_class: Enum) -> Enum:
    for type_ in enum_class:
        if id == type_.name.lower():
            return type_
    raise ValueError(f"Unknown id {id} for {enum_class}.")
