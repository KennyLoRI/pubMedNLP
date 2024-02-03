from kedro.pipeline import Pipeline, pipeline, node
from .nodes import eval_list


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=eval_list,
                inputs=["query_list", "params:modelling_params", "params:top_k_params"],
                outputs="eval_llm_response",
                name="eval_list_node",
            )
        ]
    )
