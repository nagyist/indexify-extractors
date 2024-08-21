from typing import List, Dict

from indexify_extractor_sdk.embedding.base_embedding import BaseEmbeddingExtractor
from indexify_extractor_sdk.embedding.sentence_transformer import (
    SentenceTransformersEmbedding,
)
from indexify.extractor_sdk.extractor import EmbeddingSchema


class MiniLML6Extractor(BaseEmbeddingExtractor):
    name = "tensorlake/minilm-l6"
    description = "MiniLM-L6 Sentence Transformer"
    system_dependencies = []
    embedding_indexes: Dict[str, EmbeddingSchema] = {"embedding": EmbeddingSchema(dim=384)}

    def __init__(self):
        super(MiniLML6Extractor, self).__init__(max_context_length=128)
        self._model = SentenceTransformersEmbedding(model_name="all-MiniLM-L6-v2")

    def extract_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._model.embed_ctx(texts)


if __name__ == "__main__":
    MiniLML6Extractor().extract_sample_input()
