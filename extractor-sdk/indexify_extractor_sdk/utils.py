from itertools import islice
import requests
import platform
import fsspec
import json

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
        requests.post(
            "https://getindexify.ai/api/analytics", json={"event": event, "value": value, "platform":platform.platform(), "machine": platform.machine()}
        , timeout=1)
    except Exception as e:
        # fail silently
        pass

def read_extractors_json_file():
    file_path = f's3://indexifyextractors/indexify-extractors/extractors.json'

    fs = fsspec.filesystem('s3', anon=True)

    with fs.open(file_path, "r") as file:
        # Load the JSON content from the file
        json_content = json.load(file)

    return json_content

def extractors_by_name():
    extractors_info_list = read_extractors_json_file()
    result = {}
    for extractor_info in extractors_info_list:
        result[extractor_info["name"]] = extractor_info
    return result