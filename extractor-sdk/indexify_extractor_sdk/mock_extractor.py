from indexify.extractor_sdk import Extractor, Content, Feature, EmbeddingSchema

from typing import List, Tuple, Dict

import json
from pydantic import BaseModel


class InputParams(BaseModel):
    a: int = 0
    b: str = ""


class MockExtractor(Extractor):
    name = "mock_extractor"
    input_mime_types = ["text/plain", "application/pdf", "image/jpeg"]

    system_dependencies = ["sl", "cowsay"]  # some really smalll packages for testing
    python_dependencies = ["tinytext", "pyfiglet"]
    embeddings = {"embedding": EmbeddingSchema(dim=3, distance="euclidean")}

    def __init__(self):
        super().__init__()

    def extract(self, content: Content, params: InputParams) -> List[Content]:
        return [
            Content.from_text(
                text="Hello World",
                features=[
                    Feature.embedding(values=[1, 2, 3]),
                    Feature.metadata(json.dumps({"a": 1, "b": "foo"})),
                ],
            ),
            Content.from_text(
                text="Pipe Baz",
                features=[Feature.embedding(values=[1, 2, 3])],
            ),
        ]

    def sample_input(self) -> Tuple[Content, InputParams]:
        return (
            Content.from_text("hello world"),
            InputParams(a=5, b="h").model_dump_json(),
        )


class MockExtractorWithBatch(Extractor):
    name = "mock_extractor"
    input_mime_types = ["text/plain", "application/pdf", "image/jpeg"]

    system_dependencies = ["sl", "cowsay"]  # some really smalll packages for testing
    python_dependencies = ["tinytext", "pyfiglet"]

    def __init__(self):
        super().__init__()

    def extract(self, content: Content, params: InputParams) -> List[Content]:
        pass

    def extract_batch(
        self, content_list: List[Content], input_params: List[InputParams]
    ) -> List[Content]:
        out = []
        for idx, _ in enumerate(content_list):
            out.append(
                [
                    Content.from_text(
                        text=f"Batch Result {idx + 1}",
                        features=[
                            Feature.embedding(values=[1, 2, 3]),
                            Feature.metadata(json.dumps({"a": 1, "b": "foo"})),
                        ],
                    )
                ]
            )
        return out

    def sample_input(self) -> Tuple[Content, InputParams]:
        return (
            Content.from_text("hello world"),
            InputParams(a=5, b="h").model_dump_json(),
        )


class MockExtractorsReturnsFeature(Extractor):
    def __init__(self):
        super().__init__()

    def extract(self, content: Content, params: InputParams) -> List[Feature]:
        return [
            Feature.embedding(values=[1, 2, 3]),
            Feature.metadata(json.loads('{"a": 1, "b": "foo"}')),
        ]

    def sample_input(self) -> Tuple[Content, InputParams]:
        return (Content.from_text("hello world"), InputParams(a=5, b="h"))


class MockExtractorNoInputParams(Extractor):
    def __init__(self):
        super().__init__()

    def extract(self, content: Content, params=None) -> List[Content]:
        return [
            Content.from_text(
                text="Hello World", features=[Feature.embedding(values=[1, 2, 3])]
            ),
            Content.from_text(
                text="Pipe Baz", features=[Feature.embedding(values=[1, 2, 3])]
            ),
        ]

    def sample_input(self) -> Content:
        return Content.from_text("hello world")
