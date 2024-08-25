import json
from typing import Callable, List, Literal

from indexify_extractor_sdk import Content, Extractor, Feature
from langchain import text_splitter
from langchain.docstore.document import Document
from pydantic import BaseModel


class ChunkExtractionInputParams(BaseModel):
    overlap: int = 0
    chunk_size: int = 100
    text_splitter: Literal["char", "recursive", "markdown", "html", "json"] = (
        "recursive"
    )
    headers_to_split_on: List[str] = []


class ChunkExtractor(Extractor):
    name = "tensorlake/chunk-extractor"
    description = "Text Chunk Extractor"
    python_dependencies = ["langchain", "lxml"]
    system_dependencies = []
    input_mime_types = ["text/plain", "application/json"]

    def __init__(self):
        super().__init__()

    def extract(
        self, content: Content, params: ChunkExtractionInputParams
    ) -> List[Content]:

        splitter = self._create_splitter(params)
        text = content.data.decode("utf-8")
        if content.content_type == "application/json":
            text = json.loads(text)
            text = {str(index): item for index, item in enumerate(text)}
        chunks = splitter(text)

        chunk_contents = []
        for chunk in chunks:
            if type(chunk) == Document:
                chunk_content = Content.from_text(chunk.page_content)
            else:
                chunk_content = Content.from_text(chunk)

            chunk_contents.append(chunk_content)

        return chunk_contents

    def _create_splitter(
        self, input_params: ChunkExtractionInputParams
    ) -> Callable[[str], List[str]]:
        if input_params.text_splitter == "recursive":
            return text_splitter.RecursiveCharacterTextSplitter(
                chunk_size=input_params.chunk_size,
                chunk_overlap=input_params.overlap,
            ).split_text
        elif input_params.text_splitter == "json":
            return text_splitter.RecursiveJsonSplitter(
                max_chunk_size=input_params.chunk_size
            ).split_text
        elif input_params.text_splitter == "char":
            return text_splitter.CharacterTextSplitter(
                chunk_size=input_params.chunk_size,
                chunk_overlap=input_params.overlap,
                separator="\n\n",
            ).split_text
        elif input_params.text_splitter == "markdown":
            return text_splitter.MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3"),
                ],
            ).split_text
        elif input_params.text_splitter == "html":
            return text_splitter.HTMLHeaderTextSplitter(
                headers_to_split_on=[
                    ("h1", "Header 1"),
                    ("h2", "Header 2"),
                    ("h3", "Header 3"),
                    ("h4", "Header 4"),
                ],
            ).split_text

    def sample_input(self) -> Content:
        return Content.from_text("This is a test string to be split into chunks")

    def extract_sample_input(self) -> List[Content]:
        input = self.sample_input()
        return self.extract(
            input,
            ChunkExtractionInputParams(
                overlap=0, chunk_size=5, text_splitter="recursive"
            ),
        )
