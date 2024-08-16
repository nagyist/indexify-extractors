import json
import os
from importlib import import_module
import logging
from indexify.extractor_sdk import Content, Feature, Extractor, ExtractorMetadata, EmbeddingSchema
from typing import (
    Dict,
    List,
    Tuple,
    Type,
    Union,
    get_type_hints,
)

import requests
from genson import SchemaBuilder
from pydantic import BaseModel, Field, Json

EXTRACTORS_PATH = os.path.join(os.path.expanduser("~"), ".indexify-extractors")
EXTRACTORS_MODULE = "indexify_extractors"
EXTRACTOR_MODULE_PATH = os.path.join(EXTRACTORS_PATH, EXTRACTORS_MODULE)


def load_extractor(name: str) -> Tuple[Extractor, Type[BaseModel]]:
    module_name, class_name = name.split(":")
    wrapper = ExtractorWrapper(module_name, class_name)
    return (wrapper._instance, wrapper._param_cls)


class ExtractorWrapper:
    def __init__(self, module_name: str, class_name: str):
        module = import_module(module_name)
        cls = getattr(module, class_name)
        self._instance: Extractor = cls()
        self._param_cls = get_type_hints(cls.extract).get("params", None)
        extract_batch = getattr(self._instance, "extract_batch", None)
        self._has_batch_extract = True if callable(extract_batch) else False

    def _param_from_json(self, param: Json) -> BaseModel:
        if self._param_cls is None:
            return {}

        param_dict = {}
        if param is not None and param != "null":
            param_dict = json.loads(param)

        try:
            return self._param_cls.model_validate(param_dict)
        except Exception as e:
            print(f"Error validating input params: {e}")

        return {}

    def extract_batch(
        self, content_list: Dict[str, Content], input_params: Dict[str, Json]
    ) -> Dict[str, List[Union[Feature, Content]]]:
        if self._has_batch_extract:
            task_ids = []
            task_contents = []
            params = []
            for task_id, content in content_list.items():
                param_instance = self._param_from_json(input_params.get(task_id, None))
                params.append(param_instance)
                task_ids.append(task_id)
                task_contents.append(content)

            try:
                result = self._instance.extract_batch(task_contents, params)
            except Exception as e:
                logging.error(f"Error extracting content: {e}")
                raise e
            out: Dict[str, List[Union[Feature, Content]]] = {}
            for i, extractor_out in enumerate(result):
                out[task_ids[i]] = extractor_out
            return out
        out = {}
        for task_id, content in content_list.items():
            param_instance = self._param_from_json(input_params.get(task_id, None))
            out[task_id] = self._instance.extract(content, param_instance)
        return out

    def describe(self) -> ExtractorMetadata:
        s_input = self._instance.sample_input()
        input_params = None
        if type(s_input) == tuple:
            (s_input, input_params) = s_input
        # Come back to this when we can support schemas based on user defined input params
        if input_params is None:
            input_params = (
                self._param_cls().model_dump_json()
                if self._param_cls is not None
                else None
            )
        outputs: Dict[str, List[Union[Feature, Content]]] = self.extract_batch(
            {"task_id": s_input},
            {"task_id": input_params},
        )
        embedding_schemas = {}
        json_schema = (
            self._param_cls.model_json_schema() if self._param_cls is not None else None
        )
        output = outputs["task_id"]
        for out in output:
            features = out.features if type(out) == Content else [out]
            for feature in features:
                if feature.feature_type == "embedding":
                    embedding_schema = EmbeddingSchema(
                        dim=len(feature.value["values"]),
                        distance=feature.value["distance"],
                    )
                    embedding_schemas[feature.name] = embedding_schema
        return ExtractorMetadata(
            name=self._instance.name,
            version=self._instance.version,
            description=self._instance.description,
            python_dependencies=self._instance.python_dependencies,
            system_dependencies=self._instance.system_dependencies,
            embedding_schemas=embedding_schemas,
            metadata_schemas={},
            input_mime_types=self._instance.input_mime_types,
            input_params=json_schema,
        )
