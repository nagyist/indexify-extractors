import asyncio
from .base_extractor import ExtractorWrapper 
from typing import Optional, Tuple, List
from .base_extractor import Content, EXTRACTOR_MODULE_PATH
import nanoid
import json
from .agent import ExtractorAgent, DEFAULT_BATCH_SIZE
import os
from . import coordinator_service_pb2 
from .metadata_store import ExtractorMetadataStore
from .utils import extractors_by_name
from .extractor_worker import ExtractorWorker


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
    extractor_worker = ExtractorWorker(metadata_store, workers)
    extractors: List[coordinator_service_pb2.Extractor] = []
    extractor_metadata = metadata_store.all_extractor_metadata()

    for metadata in extractor_metadata:
        embedding_schemas = {}
        for name, embedding_schema in metadata.embedding_schemas.items():
            embedding_schemas[name] = embedding_schema.model_dump_json()

        metadata_schemas = {}
        for name, metadata_schema in metadata.metadata_schemas.items():
            metadata_schemas[name] = json.dumps(metadata_schema)

        input_params = json.dumps(metadata.input_params) if metadata.input_params else json.dumps({})
        extractors.append(
            coordinator_service_pb2.Extractor(
                name=metadata.name,
                description=metadata.description,
                input_params=input_params,
                input_mime_types=metadata.input_mime_types,
                metadata_schemas=metadata_schemas,
                embedding_schemas=embedding_schemas,
            )
        )

    id = nanoid.generate()
    print(f"executor id: {id}")

    server = ExtractorAgent(
        id,
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

    # Describe the extractor.
    module, cls = extractor.split(":")
    module_name = f"indexify_extractors.{parent_dir}.{module}"
    wrapper = ExtractorWrapper(module_name, cls)
    # FIX ME - This doesn't work on Mac
    # Meant to only work on Ubuntu
    if install_system_dependencies:
        install_system_dependencies = wrapper._instance.system_dependencies
        os.system(f"sudo apt-get install -y {' '.join(install_system_dependencies)}")
    description = wrapper.describe()

    # Create a new extractor description.
    extractor_id = f"{parent_dir}.{module}:{cls}"
    create_extractor_db()
    save_extractor_description(extractor_id, description)

    print("extractor ready for testing. Run: indexify-extractor join-server")
    print(
        f"The module name for the extractor is: indexify_extractors.{parent_dir}.{module}:{cls}"
    )
    print(
        f"To package the extractor in a docker container: indexify-extractor package indexify_extractors.{parent_dir}.{module}:{cls}"
    )
