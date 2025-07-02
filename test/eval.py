from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact
from inspect_ai.solver import generate
from inspect_ai.dataset import csv_dataset

@task
def hello_world():
    return Task(
        dataset=csv_dataset("data.csv"),
        solver=[generate()],
        scorer=exact(),
    )