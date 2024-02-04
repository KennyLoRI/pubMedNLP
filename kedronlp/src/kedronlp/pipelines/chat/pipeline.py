from kedro.pipeline import Pipeline, pipeline, node
from .nodes import chat_loop


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=chat_loop,
                inputs=["params:modelling_params", "params:top_k_params"],
                outputs=None,
                name="chat_loop_node",
            )
        ]
    )
