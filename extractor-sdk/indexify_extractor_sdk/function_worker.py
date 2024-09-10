import asyncio
import concurrent
from concurrent.futures.process import BrokenProcessPool
from typing import Dict, List

from indexify.functions_sdk.indexify_functions import IndexifyFunctionWrapper
from indexify.functions_sdk.data_objects import BaseData
from indexify.functions_sdk.graph import Graph


function_wrapper_map: Dict[str, IndexifyFunctionWrapper] = {}

def _load_function(namespace: str, graph_name: str, fn_name: str, code_path: str):
    """Load an extractor to the memory: extractor_wrapper_map."""
    global function_wrapper_map
    key = f"{namespace}/{graph_name}/{fn_name}"
    if key in function_wrapper_map:
        return
    graph = Graph.from_path(code_path)
    function_wrapper = graph.get_function(fn_name)
    function_wrapper_map[key] = function_wrapper


class FunctionWorker:
    def __init__(self, workers: int = 1) -> None:
        self._executor: concurrent.futures.ProcessPoolExecutor = (
            concurrent.futures.ProcessPoolExecutor(max_workers=workers)
        )

    async def async_submit(
        self,
        namespace: str,
        graph_name: str,
        fn_name: str,
        input: BaseData,
        code_path: str,
    ) -> List[BaseData]:
        try:
            resp = await asyncio.get_running_loop().run_in_executor(
                self._executor,
                _run_function,
                namespace,
                graph_name,
                fn_name,
                input,
                code_path,
            )
        except BrokenProcessPool as mp:
            self._executor.shutdown(wait=True, cancel_futures=True)
            raise mp
        return resp

    def shutdown(self):
        self._executor.shutdown(wait=True, cancel_futures=True)


def _run_function(
    namespace: str,
    graph_name: str,
    fn_name: str,
    input: BaseData,
    code_path: str,
) -> List[BaseData]:
    key = f"{namespace}/{graph_name}/{fn_name}"
    if key not in function_wrapper_map:
        _load_function(namespace, graph_name, fn_name, code_path)

    function_wrapper: IndexifyFunctionWrapper = function_wrapper_map[key]
    results = function_wrapper.run(input)
    return results
