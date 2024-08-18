import concurrent.futures
from typing import List, Dict, Any
from .base_extractor import ExtractorWrapper
from pydantic import Json
import concurrent
from .downloader import ExtractorMetadataStore
from indexify.extractor_sdk import ExtractorMetadata
import asyncio
from concurrent.futures.process import BrokenProcessPool


# str here is ExtractorDescription.name
extractor_wrapper_map: Dict[str, ExtractorWrapper] = {}

# List of ExtractorDescription
# This is used to report the available extractors to the coordinator
extractor_descriptions: Dict[str, ExtractorMetadata] = {}

def load_extractors(name: str, extractor_module_class: str):
    """Load an extractor to the memory: extractor_wrapper_map."""
    global extractor_wrapper_map
    if name in extractor_wrapper_map:
        return
    extractor_id = f"indexify_extractors.{extractor_module_class}"
    extractor_wrapper = ExtractorWrapper.from_name(extractor_id)
    extractor_wrapper_map[name] = extractor_wrapper


class ExtractorWorker:
    def __init__(self, extractor_metadata_store: ExtractorMetadataStore, workers: int=1) -> None:
        self._extractor_metadata_store = extractor_metadata_store
        self._extractors = {}
        self._executor: concurrent.futures.ProcessPoolExecutor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)

    def submit(self, extractor: str, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, List[Any]]:
        extractor_module_class = self._extractor_metadata_store.extractor_module_class(extractor)
        return self._executor.submit(_extract_content, extractor, extractor_module_class, inputs, params, extractor)
    
    async def async_submit(self, extractor: str, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, List[Any]]:
        extractor_module_class = self._extractor_metadata_store.extractor_module_class(extractor)
        try:
            resp = await asyncio.get_running_loop().run_in_executor(
            self._executor,
            _extract_content,
            extractor,
            extractor_module_class,
            inputs,
            params,
            extractor,
            )
        except BrokenProcessPool as mp:
            self._executor.shutdown(wait=True, cancel_futures=True)
            raise mp
        return resp
    
    def shutdown(self):
        self._executor.shutdown(wait=True, cancel_futures=True)

def _extract_content(
        extractor: str,
        extractor_module_class: str,
        inputs: Dict[str, Any],
        params: Dict[str, Json],
    ) -> Dict[str, List[Any]]:
    if extractor not in extractor_wrapper_map:
        load_extractors(extractor, extractor_module_class)
        
    extractor_wrapper: ExtractorWrapper = extractor_wrapper_map[extractor]
    task_ids, extractor_inputs, extractor_params = [], [], []
    for task_id, content in inputs.items():
        task_ids.append(task_id)
        extractor_inputs.append(content)
        extractor_params.append(params[task_id])

    results = extractor_wrapper.extract_batch(inputs, params)
    output = {}
    for (task_id, output) in zip(task_ids, results):
        output[task_id] = output
    return output
