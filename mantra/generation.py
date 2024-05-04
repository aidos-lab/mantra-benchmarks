import numpy as np
from sklearn.model_selection import train_test_split

from mantra.random import get_common_random_state
from mantra.transforms import NameToClassTransform


def generate_random_split(
    triangulations,
    task_type,
    test_size=0.2,
):
    indices_dataset = np.arange(len(triangulations))
    match task_type:
        case "orientability":
            labels = np.array(
                [
                    int(triangulation["orientable"])
                    for triangulation in triangulations
                ]
            )
        case "betti_numbers":
            # For betti numbers, we do not stratify the split, as there are many different
            # configurations for the three betti numbers.
            labels = None
        case "name":
            get_label_from_name = NameToClassTransform().class_dict
            labels = np.array(
                [
                    get_label_from_name[triangulation["name"]]
                    for triangulation in triangulations
                ]
            )
        case _:
            raise NotImplementedError("Unknown task type")
    X_train, X_test = train_test_split(
        indices_dataset,
        test_size=test_size,
        shuffle=True,
        stratify=labels,
        random_state=get_common_random_state(),
    )
    return np.array(X_train), np.array(X_test)
