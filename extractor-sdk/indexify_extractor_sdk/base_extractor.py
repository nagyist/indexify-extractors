import inspect
import logging
import os
from importlib import import_module
from typing import Any, Dict, List, Optional, Type, Union, get_type_hints

from indexify.extractor_sdk import Content, Extractor, ExtractorMetadata, Feature
from pydantic import BaseModel, Json, create_model


class ExtractorPayload(BaseModel):
    data: bytes
    content_type: str
    extract_args: Optional[Dict] = None
    class_args: Optional[Dict] = None


EXTRACTORS_PATH = os.path.join(os.path.expanduser("~"), ".indexify-extractors")
EXTRACTORS_MODULE = "indexify_extractors"
EXTRACTOR_MODULE_PATH = os.path.join(EXTRACTORS_PATH, EXTRACTORS_MODULE)


class ExtractorPayload(BaseModel):
    data: bytes
    content_type: str
    extract_args: Optional[Dict] = None
    class_args: Optional[Dict] = None


def create_pydantic_model_from_class_init_args(cls):
    signature = inspect.signature(cls.__init__)

    type_hints = get_type_hints(cls.__init__)

    fields = {}
    for param_name, param in signature.parameters.items():
        # Skip 'self' parameter
        if param_name == "self":
            continue
        param_type = type_hints.get(
            param_name,
            type(param.default) if param.default is not param.empty else None,
        )
        if param_type is None:
            param_type = Any
        default_value = ... if param.default is param.empty else param.default
        fields[param_name] = (param_type, default_value)

    return create_model(f"{cls.__name__}Model", **fields)


class ExtractorWrapper:
    def __init__(self, module_name: str, class_name: str):
        module = import_module(module_name)
        self._cls: Type[Extractor] = getattr(module, class_name)
        self._instance: Extractor = None
        self._class_args: Type[BaseModel] = create_pydantic_model_from_class_init_args(
            self._cls
        )
        extract_batch = getattr(self._cls, "extract_batch", None)
        self._has_batch_extract = True if callable(extract_batch) else False
        self._extractor_args_cls: Type[BaseModel] = get_type_hints(
            self._cls.extract
        ).get("params", None)

    @classmethod
    def from_name(cls, full_module_name: str):
        module_name, class_name = full_module_name.split(":")
        return cls(module_name, class_name)

    def extract_batch(
        self, content_list: Dict[str, ExtractorPayload]
    ) -> Dict[str, List[Union[Feature, Content]]]:
        if self._instance is None:
            self._instance = self._cls()
        task_ids = []
        task_contents = []
        args = []
        out: Dict[str, List[Union[Feature, Content]]] = {}
        for task_id, payload in content_list.items():
            content = Content(
                id=None,
                data=payload.data,
                content_type=payload.content_type,
                features=[],
            )
            extractor_args = None
            if self._extractor_args_cls:
                if payload.extract_args:
                    extractor_args = self._extractor_args_cls.model_validate(
                        payload.extract_args
                    )
                else:
                    extractor_args = self._extractor_args_cls()
            args.append(extractor_args)
            task_ids.append(task_id)
            task_contents.append(content)
        if self._has_batch_extract:
            try:
                result = self._instance.extract_batch(task_contents, args)
            except Exception as e:
                logging.error(f"Error extracting content: {e}")
                raise e
            for i, extractor_out in enumerate(result):
                out[task_ids[i]] = extractor_out
            return out
        for task_id, content, extractor_args in zip(task_ids, task_contents, args):
            out[task_id] = self._instance.extract(content, extractor_args)
        return out

    def describe(self) -> ExtractorMetadata:
        embeddings_schemas = {}
        for name, embedding_schema in self._cls.embeddings.items():
            embeddings_schemas[name] = embedding_schema.model_dump()
        return ExtractorMetadata(
            name=self._cls.name,
            version=self._cls.version,
            description=self._cls.description,
            python_dependencies=self._cls.python_dependencies,
            system_dependencies=self._cls.system_dependencies,
            embedding_schemas=embeddings_schemas,
            metadata_schemas={},
            input_mime_types=self._cls.input_mime_types,
            input_params=(
                self._extractor_args_cls.model_json_schema()
                if self._extractor_args_cls
                else None
            ),
        )
