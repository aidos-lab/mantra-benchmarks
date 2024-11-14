from collections import Counter
from typing import List, Tuple

import numpy as np

from datasets.transforms import NAME_TO_CLASS
from metrics.tasks import TaskType


def convert_to_list(imb: Counter) -> Tuple[List[float], List[float]]:
    """Handling of Counter collection"""
    return list(imb.keys()), list(imb.values())


def derive_weights(imbalance: List[float]):
    imbalance = np.asarray(imbalance)
    inversed_imbalance = np.sum(imbalance) / (imbalance * len(imbalance))
    return inversed_imbalance


def normalise_imbalance(imbalance: List[float]) -> List[float]:
    tmp = np.asarray(imbalance)
    return list(tmp / tmp.sum())


def sort_imbalance(
        imbalance: List[float], keys: List, task_type: TaskType
) -> List[float]:
    """
    Sorts the imbalance counters by the index the neural networks intend to predict.

    Orientability: Only the two classes False and True, hence no sort necessary.
    Betti numbers: No imbalance handling implemented, hence no logic necessary
    Name classification: Use NAME_TO_CLASS dict from datasets.transforms
    """
    if task_type == TaskType.BETTI_NUMBERS:
        raise ValueError(
            "No imbalance handling for the betti numbers task intended."
        )

    if task_type == TaskType.NAME:
        sorted_ = [0 for i in keys]
        for i in range(len(keys)):
            k = keys[i]
            v = imbalance[i]
            sorted_[NAME_TO_CLASS[k]] = v
        return sorted_

    return imbalance


def sorted_imbalance_weights(
        imbalance: Counter, task_type: TaskType
) -> List[float]:
    """
    Returns normalised sorted inverse imbalance.

    E.g. if there are 90 samples of class 0 and 10 samples of class 1, it will return [0.1, 0.9]
    """
    imb_keys, imb_values = convert_to_list(imbalance)
    imbalance = sort_imbalance(imb_values, imb_keys, task_type)
    inversed_imbalance = derive_weights(imbalance)
    norm_imbalanced = normalise_imbalance(inversed_imbalance)
    return norm_imbalanced
