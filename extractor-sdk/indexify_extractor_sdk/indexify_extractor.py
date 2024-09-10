import asyncio
import json
import os
from typing import List, Optional, Tuple

import nanoid

from .agent import DEFAULT_BATCH_SIZE, ExtractorAgent
from .base_extractor import EXTRACTOR_MODULE_PATH, Content, ExtractorWrapper
from .extractor_worker import ExtractorWorker
from .metadata_store import ExtractorMetadataStore
from .server_if import coordinator_service_pb2


def local(extractor: str, text: Optional[str] = None, file: Optional[str] = None):
    if text and file:
        raise ValueError("You can only pass either text or file, not both.")
    if not text and not file:
        raise ValueError("You need to pass either text or file")
    if text:
        content = Content.from_text(text)
    if file:
        content = Content.from_file(file)
    module, cls = extractor.split(":")
    wrapper = ExtractorWrapper(module, cls)
    result = wrapper.extract_batch({"task_id": content}, input_params={"task_id": "{}"})
    print(result)


def split_validate_extractor(name: str) -> Tuple[str, str]:
    try:
        module, cls = name.split(":")
    except ValueError:
        raise ValueError(
            "The extractor name should be in the format 'module_name:class_name'"
        )
    return module, cls


def join(
    workers: int,
    listen_port: int,
    coordinator_addr: str = "localhost:8950",
    ingestion_addr: str = "localhost:8900",
    advertise_addr: Optional[str] = None,
    config_path: Optional[str] = None,
    download_method: str = "direct",
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    print(
        f"joining {coordinator_addr} and sending extracted content to {ingestion_addr}"
    )
    metadata_store = ExtractorMetadataStore()
    extractor_worker = ExtractorWorker(workers)
    extractors: List[coordinator_service_pb2.Extractor] = []

    id = nanoid.generate()
    print(f"executor id: {id}")

    server = ExtractorAgent(
        id,
        metadata_store=metadata_store,
        extractors=extractors,
        extractor_worker=extractor_worker,
        coordinator_addr=coordinator_addr,
        num_workers=workers,
        listen_port=listen_port,
        ingestion_addr=ingestion_addr,
        advertise_addr=advertise_addr,
        config_path=config_path,
        download_method=download_method,
        batch_size=batch_size,
    )

    try:
        asyncio.get_event_loop().run_until_complete(server.run())
    except asyncio.CancelledError as ex:
        print("exiting gracefully", ex)


def describe_sync(extractor_name: str):
    metadata_store = ExtractorMetadataStore()
    module_class_name = metadata_store.extractor_module_class(extractor_name)
    description = ExtractorWrapper.from_name(module_class_name).describe()
    print(description)


def install_local(extractor, install_system_dependencies=False):
    # Copy everything in the current directory to the extractors directory.
    parent_dir = os.path.basename(os.getcwd())
    destination = os.path.join(EXTRACTOR_MODULE_PATH, parent_dir)
    os.system(f"cp -r . {destination}")
    print(f"copied to {destination} for testing")
    metadata_store = ExtractorMetadataStore()

    # Describe the extractor.
    module, cls = extractor.split(":")
    module_name = f"indexify_extractors.{parent_dir}.{module}"
    wrapper = ExtractorWrapper(module_name, cls)
    # FIX ME - This doesn't work on Mac
    # Meant to only work on Ubuntu
    if install_system_dependencies:
        install_system_dependencies = wrapper.describe().system_dependencies
        os.system(f"sudo apt-get install -y {' '.join(install_system_dependencies)}")
    description = wrapper.describe()

    # Create a new extractor description.
    extractor_id = f"{parent_dir}.{module}:{cls}"
    metadata_store.save_description(extractor_id, description)

    print("extractor ready for testing. Run: indexify-extractor join-server")
    print(
        f"The module name for the extractor is: indexify_extractors.{parent_dir}.{module}:{cls}"
    )
    print(
        f"To package the extractor in a docker container: indexify-extractor package indexify_extractors.{parent_dir}.{module}:{cls}"
    )
