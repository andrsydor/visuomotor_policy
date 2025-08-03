from typing import Type

from visuomotor.dataset.dataset_base import DatasetBase
from visuomotor.dataset.push_t_dataset import PushTImageDataset


class Task:
    @staticmethod
    def dataset_class() -> Type[DatasetBase]:
        raise NotImplementedError("Should be implemented in child")
    

class PushTTask(Task):
    @staticmethod
    def dataset_class():
        return PushTImageDataset
    

TASKS = {
    "push_t": PushTTask
}


def task_from_string(task_name: str) -> Task:
    task_class = TASKS.get(task_name, None)
    
    if not task_class:
        raise NotImplementedError("Task not implemented")
    
    return task_class
