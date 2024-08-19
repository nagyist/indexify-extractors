import asyncio
import concurrent
import concurrent.futures
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Dict, List

from indexify.extractor_sdk import ExtractorMetadata

from .base_extractor import ExtractorPayload, ExtractorWrapper

# str here is ExtractorDescription.name
extractor_wrapper_map: Dict[str, ExtractorWrapper] = {}

# List of ExtractorDescription
# This is used to report the available extractors to the coordinator
extractor_descriptions: Dict[str, ExtractorMetadata] = {}


def _load_extractors(name: str, extractor_module_class: str):
    """Load an extractor to the memory: extractor_wrapper_map."""
    global extractor_wrapper_map
    if name in extractor_wrapper_map:
        return
    extractor_wrapper = ExtractorWrapper.from_name(extractor_module_class)
    extractor_wrapper_map[name] = extractor_wrapper


class ExtractorWorker:
    def __init__(self, workers: int = 1) -> None:
        self._executor: concurrent.futures.ProcessPoolExecutor = (
            concurrent.futures.ProcessPoolExecutor(max_workers=workers)
        )

    async def async_submit(
        self,
        extractor: str,
        extractor_module_class: str,
        inputs: Dict[str, ExtractorPayload],
    ) -> Dict[str, List[Any]]:
        try:
            resp = await asyncio.get_running_loop().run_in_executor(
                self._executor,
                _extract_content,
                extractor,
                extractor_module_class,
                inputs,
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
    inputs: Dict[str, ExtractorPayload],
) -> Dict[str, List[Any]]:
    # TODO Use the hash of Class Args and Extractor Name for caching extractors
    if extractor not in extractor_wrapper_map:
        _load_extractors(extractor, extractor_module_class)

    extractor_wrapper: ExtractorWrapper = extractor_wrapper_map[extractor]
    task_ids, extractor_inputs = [], []
    for task_id, input in inputs.items():
        task_ids.append(task_id)
        extractor_inputs.append(input)

    results = extractor_wrapper.extract_batch(inputs)
    return results
