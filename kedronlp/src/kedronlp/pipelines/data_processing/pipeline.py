from kedro.pipeline import Pipeline, node, pipeline
from .nodes import extract_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=extract_data,
                inputs=None,
                outputs="extract_data",
                name="extract_data_node",
            ),
        ]
    )
