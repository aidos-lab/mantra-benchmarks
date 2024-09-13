from typing import List, Callable, Dict
from enum import Enum

from datasets.transforms import (
    orientability_transforms,
    name_transforms,
    betti_numbers_transforms_2manifold,
    betti_numbers_transforms_3manifold,
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
    MetricTrainValTest,
)

from enum import Enum
from torch_geometric.transforms import Compose
from datasets.dataset_types import DatasetType


class Task:
    transforms: Compose
    loss_fn: Callable
    get_metrics: Callable[[], MetricTrainValTest]
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
    def __init__(
        self, transforms: List[Callable], ds_type: DatasetType
    ) -> None:
        betti_tr = (
            betti_numbers_transforms_3manifold
            if ds_type == DatasetType.FULL_3D
            else betti_numbers_transforms_2manifold
        )
        super().__init__(
            transforms=Compose(transforms + betti_tr),
            loss_fn=betti_loss_fn,
            metrics=get_betti_numbers_metrics,
            accuracies=compute_betti_numbers_accuracies,
        )


class TaskType(Enum):
    NAME = "name"
    ORIENTABILITY = "orientability"
    BETTI_NUMBERS = "betti_numbers"


def get_task_lookup(
    transforms: List[Callable], ds_type: DatasetType
) -> Dict[TaskType, Task]:
    res: Dict[TaskType, Task] = {
        TaskType.NAME: NameTask(transforms),
        TaskType.ORIENTABILITY: OrientabilityTask(transforms),
        TaskType.BETTI_NUMBERS: BettiNumbersTask(transforms, ds_type=ds_type),
    }

    return res


class_transforms_lookup_2manifold: Dict[TaskType, List[Callable]] = {
    TaskType.BETTI_NUMBERS: betti_numbers_transforms_2manifold,
    TaskType.ORIENTABILITY: orientability_transforms,
    TaskType.NAME: name_transforms,
}

class_transforms_lookup_3manifold: Dict[TaskType, List[Callable]] = {
    TaskType.BETTI_NUMBERS: betti_numbers_transforms_3manifold,
    TaskType.ORIENTABILITY: orientability_transforms,
}
