import unittest
from typing import List, Optional

from indexify.extractor_sdk import (Content, EmbeddingSchema,
                                    ExtractorMetadata, Feature)
from indexify_extractor_sdk.base_extractor import (
    ExtractorPayload, ExtractorWrapper,
    create_pydantic_model_from_class_init_args)
from indexify_extractor_sdk.mock_extractor import InputParams
from pydantic import BaseModel


class TestCreatePydanticModelFromClass(unittest.TestCase):
    def test_simple_class(self):
        class Person:
            def __init__(self, name: str, age: int):
                self.name = name
                self.age = age

        PersonModel = create_pydantic_model_from_class_init_args(Person)
        self.assertTrue(issubclass(PersonModel, BaseModel))
        self.assertEqual(set(PersonModel.model_fields.keys()), {"name", "age"})
        self.assertEqual(
            PersonModel.model_json_schema(),
            {
                "properties": {
                    "age": {"title": "Age", "type": "integer"},
                    "name": {"title": "Name", "type": "string"},
                },
                "required": ["name", "age"],
                "title": "PersonModel",
                "type": "object",
            },
        )

    def test_class_with_default_values(self):
        class Car:
            def __init__(self, brand: str, model: str, year: int = 2023):
                self.brand = brand
                self.model = model
                self.year = year

        CarModel = create_pydantic_model_from_class_init_args(Car)
        self.assertTrue(issubclass(CarModel, BaseModel))
        self.assertEqual(set(CarModel.model_fields.keys()), {"brand", "model", "year"})
        self.assertEqual(CarModel.model_fields["year"].default, 2023)

    def test_class_with_optional_types(self):
        class Book:
            def __init__(self, title: str, author: str, pages: Optional[int] = None):
                self.title = title
                self.author = author
                self.pages = pages

        BookModel = create_pydantic_model_from_class_init_args(Book)
        self.assertTrue(issubclass(BookModel, BaseModel))
        self.assertEqual(
            set(BookModel.model_fields.keys()), {"title", "author", "pages"}
        )
        self.assertEqual(
            BookModel.model_json_schema(),
            {
                "properties": {
                    "title": {"title": "Title", "type": "string"},
                    "author": {"title": "Author", "type": "string"},
                    "pages": {
                        "anyOf": [{"type": "integer"}, {"type": "null"}],
                        "default": None,
                        "title": "Pages",
                    },
                },
                "required": ["title", "author"],
                "title": "BookModel",
                "type": "object",
            },
        )

    def test_class_with_complex_types(self):
        class Library:
            def __init__(self, name: str, books: List[str], capacity: int = 1000):
                self.name = name
                self.books = books
                self.capacity = capacity

        LibraryModel = create_pydantic_model_from_class_init_args(Library)
        self.assertTrue(issubclass(LibraryModel, BaseModel))
        self.assertEqual(
            set(LibraryModel.model_fields.keys()), {"name", "books", "capacity"}
        )
        self.assertEqual(
            LibraryModel.model_json_schema(),
            {
                "properties": {
                    "books": {
                        "items": {"type": "string"},
                        "title": "Books",
                        "type": "array",
                    },
                    "capacity": {
                        "default": 1000,
                        "title": "Capacity",
                        "type": "integer",
                    },
                    "name": {"title": "Name", "type": "string"},
                },
                "required": ["name", "books"],
                "title": "LibraryModel",
                "type": "object",
            },
        )

    def test_class_without_type_hints(self):
        class NoHints:
            def __init__(self, a, b, c=10):
                self.a = a
                self.b = b
                self.c = c

        NoHintsModel = create_pydantic_model_from_class_init_args(NoHints)
        self.assertTrue(issubclass(NoHintsModel, BaseModel))
        self.assertEqual(set(NoHintsModel.model_fields.keys()), {"a", "b", "c"})
        self.assertEqual(NoHintsModel.model_fields["c"].default, 10)


class TestExtractorWrapper(unittest.TestCase):
    def setUp(self):
        self.mock_extractor_wrapper = ExtractorWrapper.from_name(
            "indexify_extractor_sdk.mock_extractor:MockExtractor"
        )
        self.batched_mock_extractor_wrapper = ExtractorWrapper.from_name(
            "indexify_extractor_sdk.mock_extractor:MockExtractorWithBatch"
        )
        self.mock_extractor_returns_feature_wrapper = ExtractorWrapper.from_name(
            "indexify_extractor_sdk.mock_extractor:MockExtractorsReturnsFeature"
        )
        self.mock_extractor_no_input_params_wrapper = ExtractorWrapper.from_name(
            "indexify_extractor_sdk.mock_extractor:MockExtractorNoInputParams"
        )

    def test_extract_batch(self):
        result = self.mock_extractor_wrapper.extract_batch(
            {
                "task1": ExtractorPayload(
                    data=b"test content",
                    content_type="text/plain",
                    extract_args={"a": 1, "b": "test"},
                )
            }
        )

        self.assertIn("task1", result)
        self.assertEqual(len(result["task1"]), 2)
        self.assertIsInstance(result["task1"][0], Content)
        self.assertIsInstance(result["task1"][1], Content)
        self.assertEqual(result["task1"][0].data, b"Hello World")
        self.assertEqual(result["task1"][1].data, b"Pipe Baz")

    def test_extract_batch_returns_feature(self):
        result = self.mock_extractor_returns_feature_wrapper.extract_batch(
            {
                "task1": ExtractorPayload(
                    data=b"test content",
                    content_type="text/plain",
                    extract_args={"a": 1, "b": "test"},
                )
            }
        )

        self.assertIn("task1", result)
        self.assertEqual(len(result["task1"]), 2)
        self.assertIsInstance(result["task1"][0], Feature)
        self.assertIsInstance(result["task1"][1], Feature)
        self.assertEqual(result["task1"][0].feature_type, "embedding")
        self.assertEqual(result["task1"][1].feature_type, "metadata")

    def test_extract_batch_no_input_params(self):
        result = self.mock_extractor_no_input_params_wrapper.extract_batch(
            {"task1": ExtractorPayload(data=b"test content", content_type="text/plain")}
        )

        self.assertIn("task1", result)
        self.assertEqual(len(result["task1"]), 2)
        self.assertIsInstance(result["task1"][0], Content)
        self.assertIsInstance(result["task1"][1], Content)
        self.assertEqual(result["task1"][0].data, b"Hello World")
        self.assertEqual(result["task1"][1].data, b"Pipe Baz")

    def test_describe(self):
        metadata = self.mock_extractor_wrapper.describe()

        self.assertIsInstance(metadata, ExtractorMetadata)
        self.assertEqual(metadata.name, "mock_extractor")
        self.assertEqual(
            metadata.input_mime_types, ["text/plain", "application/pdf", "image/jpeg"]
        )
        self.assertEqual(metadata.system_dependencies, ["sl", "cowsay"])
        self.assertEqual(metadata.python_dependencies, ["tinytext", "pyfiglet"])
        self.assertIn("embedding", metadata.embedding_schemas)
        self.assertIsInstance(metadata.embedding_schemas["embedding"], EmbeddingSchema)
        self.assertEqual(metadata.embedding_schemas["embedding"].dim, 3)

    def test_describe_no_input_params(self):
        metadata = self.mock_extractor_no_input_params_wrapper.describe()

        self.assertIsInstance(metadata, ExtractorMetadata)
        self.assertIsNone(metadata.input_params)

    def test_extract_batch_with_batch_method(self):
        input_params = InputParams(a=1, b="test").model_dump()

        result = self.batched_mock_extractor_wrapper.extract_batch(
            {
                "task1": ExtractorPayload(
                    data=b"test content 1",
                    content_type="text/plain",
                    extract_args=input_params,
                ),
                "task2": ExtractorPayload(
                    data=b"test content 2",
                    content_type="text/plain",
                    extract_args=input_params,
                ),
            }
        )

        self.assertIn("task1", result)
        self.assertIn("task2", result)
        self.assertEqual(result["task1"][0].data, b"Batch Result 1")
        self.assertEqual(result["task2"][0].data, b"Batch Result 2")


if __name__ == "__main__":
    unittest.main()
