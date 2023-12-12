from kedro.pipeline import Pipeline, node, pipeline
from .nodes import extract_data, process_extract_data, get_user_query


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=extract_data,
                inputs="params:extract_params",
                outputs="extract_data",
                name="extract_data_node",
            ),
            node(
                func=process_extract_data,
                inputs="extract_data",
                outputs="paragraphed_data",
                name="process_data_node",
            ),
            node(
                func=get_user_query,
                inputs=None,
                outputs="user_query",
                name="get_user_query_node",
            ),
        ]
    )
