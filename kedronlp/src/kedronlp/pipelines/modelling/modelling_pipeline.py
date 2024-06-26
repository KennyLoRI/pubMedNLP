from kedro.pipeline import Pipeline, node, pipeline
from .modelling_nodes import get_user_query, modelling_answer, top_k_retrieval

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_user_query,
                inputs="params:modelling_params",
                outputs=["user_query", "user_query_filters"],
                name="get_user_query_node",
            ),
            node(
                func=top_k_retrieval,
                inputs=["user_query", "params:top_k_params","params:modelling_params", "user_query_filters"],
                outputs="top_k_docs",
                name="top_k_retrieval_node",
            ),
            node(
                func=modelling_answer,
                inputs=["user_query", "top_k_docs", "params:modelling_params"],
                outputs="llm_response",
                name="modelling_answer_node"
            )

        ]
    )