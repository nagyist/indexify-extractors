import json
import platform
from itertools import islice

import fsspec
import httpx 
from rich.console import Console

console = Console()

# https://docs.python.org/3/library/itertools.html#itertools.batched
def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def log_event(event, value):
    try:
        httpx.post(
            "https://getindexify.ai/api/analytics",
            json={
                "event": event,
                "value": value,
                "platform": platform.platform(),
                "machine": platform.machine(),
            },
            timeout=1,
        )
    except Exception as e:
        # fail silently
        pass


class ExtractorIndex:
    def __init__(self) -> None:
        file_path = f"s3://indexifyextractors/indexify-extractors/extractors.json"
        fs = fsspec.filesystem("s3", anon=True)
        self._index = {}
        try:
            with fs.open(file_path, "r") as file:
                # Load the JSON content from the file
                json_index = json.load(file)
                for extractor_info in json_index:
                    self._index[extractor_info["name"]] = extractor_info
        except Exception as e:
            value = {"stage": "extractor_list", "error": str(e)}
            log_event("extractor_download_failed", value=value)
            raise e

    def metadata_by_name(self, name):
        try:
            return self._index.get(name)
        except Exception as e:
            value = {"stage": "extractor_metadata", "error": str(e)}
            log_event("extractor_download_failed", value=value)
            console.print(f"[bold #f04318]Extractor {name} not found[/]")
            console.print(
                f"[bold #f04318]Use command: [yellow]indexify-extractor list[/yellow] to see the list of available extractors[/]"
            )
            raise e

    def all_metadata(self):
        return self._index
