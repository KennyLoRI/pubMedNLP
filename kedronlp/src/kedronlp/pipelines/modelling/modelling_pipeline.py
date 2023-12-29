from kedro.pipeline import Pipeline, node, pipeline
from .modelling_nodes import get_user_query, instantiate_llm, modelling_answer

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_user_query,
                inputs=None,
                outputs="user_query",
                name="get_user_query_node",
            ),
            node(
                func=modelling_answer,
                inputs="user_query",
                outputs=None,
                name="modelling_answer_node"
            )

        ]
    )