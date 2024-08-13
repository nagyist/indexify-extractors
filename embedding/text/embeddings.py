from typing import List
from pydantic import BaseModel, Field
from indexify_extractor_sdk.embedding.base_embedding import BaseEmbeddingExtractor
from sentence_transformers import SentenceTransformer

class EmbeddingConfig(BaseModel):
    model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Name of the embedding model to use")
    normalize_embeddings: bool = Field(default=True, description="Whether to normalize the embeddings")
    trust_remote_code: bool = Field(default=True, description="Whether to trust remote code when loading the model")

class EmbeddingExtractor(BaseEmbeddingExtractor):
    name = "tensorlake/embeddings"
    description = "Extractor supporting multiple embedding models"
    system_dependencies = []

    def __init__(self, config: EmbeddingConfig = EmbeddingConfig()):
        super(EmbeddingExtractor, self).__init__(max_context_length=1024)
        self.model = SentenceTransformer(config.model, trust_remote_code=config.trust_remote_code)
        self.normalize_embeddings = config.normalize_embeddings

    def extract_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=self.normalize_embeddings)

if __name__ == "__main__":
    # Example usage with default settings
    extractor = EmbeddingExtractor()
    embeddings = extractor.extract_embeddings(["What are some ways to reduce stress?", "These are the benefits of drinking green tea."])
    print("Default embeddings:", embeddings)

    # Example usage with custom settings
    custom_config = EmbeddingConfig(
        model="BAAI/bge-base-en-v1.5",
        normalize_embeddings=True,
        trust_remote_code=True
    )
    custom_extractor = EmbeddingExtractor(custom_config)
    custom_embeddings = custom_extractor.extract_embeddings(["Custom model test."])
    print("Custom embeddings:", custom_embeddings)
