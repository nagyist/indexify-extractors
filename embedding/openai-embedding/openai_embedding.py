from typing import List, Dict

from indexify_extractor_sdk.embedding.base_embedding import BaseEmbeddingExtractor
from indexify.extractor_sdk.extractor import EmbeddingSchema
from openai import OpenAI


class OpenAIEmbeddingExtractor(BaseEmbeddingExtractor):
    name = "tensorlake/openai-embedding-ada-002-extractor"
    description = "OpenAI Embedding extractor"
    python_dependencies = ["openai"]
    system_dependencies = []
    embedding_indexes: Dict[str, EmbeddingSchema] = {"embedding": EmbeddingSchema(dim=1536)}

    def __init__(self):
        super(OpenAIEmbeddingExtractor, self).__init__(max_context_length=128)
        self.model_name = "text-embedding-ada-002"
        self.client = OpenAI()

    def extract_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.client.embeddings.create(input=texts, model=self.model_name)
        return [data.embedding for data in embeddings.data]


if __name__ == "__main__":
    OpenAIEmbeddingExtractor().extract_sample_input()
