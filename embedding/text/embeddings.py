from typing import List, Literal
from pydantic import BaseModel, Field
from indexify_extractor_sdk.embedding.base_embedding import BaseEmbeddingExtractor
from sentence_transformers import SentenceTransformer

class EmbeddingConfig(BaseModel):
    model: Literal[
        "dunzhang/stella_en_400M_v5",
        "Alibaba-NLP/gte-large-en-v1.5",
        "mixedbread-ai/mxbai-embed-large-v1",
        "WhereIsAI/UAE-Large-V1",
        "avsolatorio/GIST-large-Embedding-v0",
        "BAAI/bge-large-en-v1.5",
        "Alibaba-NLP/gte-base-en-v1.5",
        "avsolatorio/GIST-Embedding-v0",
        "BAAI/bge-base-en-v1.5"
    ] = Field(default="BAAI/bge-base-en-v1.5", description="Name of the embedding model to use")

class EmbeddingExtractor(BaseEmbeddingExtractor):
    name = "tensorlake/embeddings"
    description = "Extractor supporting multiple embedding models"
    system_dependencies = []

    def __init__(self, config: EmbeddingConfig):
        super(EmbeddingExtractor, self).__init__(max_context_length=1024)
        self.model = SentenceTransformer(config.model)

    def extract_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts)

if __name__ == "__main__":
    # Example usage
    config = EmbeddingConfig(model_name="BAAI/bge-base-en-v1.5")
    EmbeddingExtractor(config).extract_embeddings(["What are some ways to reduce stress?", "These are the benefits of drinking green tea."])
