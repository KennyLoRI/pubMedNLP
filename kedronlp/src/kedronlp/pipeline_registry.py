"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from .pipelines.data_processing.pipeline import create_pipeline as data_processing_pipeline
from .pipelines.modelling.modelling_pipeline import create_pipeline as modelling_pipeline
def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = {
        "data_processing": data_processing_pipeline(),
        "modelling": modelling_pipeline()
    }
    #pipelines = find_pipelines()
    #print("Discovered pipelines:", pipelines.keys())
    pipelines["__default__"] = sum(pipelines.values())
    return pipelines
