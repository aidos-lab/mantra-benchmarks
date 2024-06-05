from typing import List, Callable, Dict, Tuple
from enum import Enum

from mantra.transforms import (
    orientability_transforms,
    name_transforms,
    betti_numbers_transforms,
)

from .accuracies import (
    compute_betti_numbers_accuracies,
    compute_name_accuracies,
    compute_orientability_accuracies,
)
from .loss import (
    name_loss_fn,
    orientability_loss_fn,
    betti_loss_fn,
)
from .metrics import (
    get_betti_numbers_metrics,
    get_name_metrics,
    get_orientability_metrics,
)

from enum import Enum, auto
from torch_geometric.transforms import Compose


class Task:
    transforms: Compose
    loss_fn: Callable
    get_metrics: Callable[[], Tuple]
    accuracies: Callable

    def __init__(self, transforms, loss_fn, metrics, accuracies) -> None:
        self.transforms = transforms
        self.loss_fn = loss_fn
        self.get_metrics = metrics
        self.accuracies = accuracies


class NameTask(Task):
    def __init__(self, transforms: List[Callable]) -> None:
        super().__init__(
            transforms=Compose(transforms + name_transforms),
            loss_fn=name_loss_fn,
            metrics=get_name_metrics,
            accuracies=compute_name_accuracies,
        )


class OrientabilityTask(Task):
    def __init__(self, transforms: List[Callable]) -> None:
        super().__init__(
            transforms=Compose(transforms + orientability_transforms),
            loss_fn=orientability_loss_fn,
            metrics=get_orientability_metrics,
            accuracies=compute_orientability_accuracies,
        )


class BettiNumbersTask(Task):
    def __init__(self, transforms: List[Callable]) -> None:
        super().__init__(
            transforms=Compose(transforms + betti_numbers_transforms),
            loss_fn=betti_loss_fn,
            metrics=get_betti_numbers_metrics,
            accuracies=compute_betti_numbers_accuracies,
        )


class TaskType(Enum):
    NAME = "name"
    ORIENTABILITY = "orientability"
    BETTI_NUMBERS = "betti_numbers"


def get_task_lookup(transforms: List[Callable]) -> Dict[TaskType, Task]:
    res: Dict[TaskType, Task] = {
        TaskType.NAME: NameTask(transforms),
        TaskType.ORIENTABILITY: OrientabilityTask(transforms),
        TaskType.BETTI_NUMBERS: BettiNumbersTask(transforms),
    }

    return res
