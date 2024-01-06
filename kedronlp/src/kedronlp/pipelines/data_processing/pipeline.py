from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    extract_data,
    create_paragraphs,
    paragraph2vec,
    vec2chroma,
)


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
                func=create_paragraphs,
                inputs="extract_data",
                outputs="paragraphs",
                name="create_paragraphs_node",
            ),
            node(
                func=paragraph2vec,
                inputs="paragraphs",
                outputs="paragraph_embeddings",
                name="paragraph2vec_node",
            ),
            node(
                func=vec2chroma,
                inputs="paragraph_embeddings",
                outputs=None,
                name="vec2chroma_node",
            ),
        ]
    )
