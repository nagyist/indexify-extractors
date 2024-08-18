import os
import sys
import subprocess

import fsspec
from rich.console import Console
from rich.panel import Panel

from .extractor_worker import ExtractorWrapper
from .base_extractor import EXTRACTORS_PATH, EXTRACTOR_MODULE_PATH
from .metadata_store import ExtractorMetadataStore
from .utils import log_event, ExtractorIndex

console = Console()

VENV_PATH = os.path.join(EXTRACTORS_PATH, "ve")


def print_instructions():
    message = (
        """To use all the downloaded extractors run the following:\n[bold #4AA4F4]"""
    )

    if not os.environ.get("VIRTUAL_ENV"):
        message += f"source {VENV_PATH}/bin/activate\n"

    message += f"indexify-extractor join-server[/]"
    console.print(
        Panel(message, title="[bold magenta]Run the extractor[/]", expand=True)
    )


def install_dependencies(directory_path):
    console.print("[bold #4AA4F4]Installing dependencies...[/]")
    requirements_path = os.path.join(directory_path, "requirements.txt")

    if not os.path.exists(requirements_path):
        raise ValueError("Unable to find requirements.txt")

    # Use sys.executable to get the path of the current Python interpreter
    python_executable = sys.executable
    pip_command = [python_executable, "-m", "pip", "install", "-r", requirements_path]

    try:
        subprocess.check_call(pip_command)
        subprocess.check_call(
            [python_executable, "-m", "pip", "install", "indexify-extractor-sdk"]
        )
    except subprocess.CalledProcessError as e:
        console.print(f"[bold #f04318]Error installing dependencies: {e}[/]")
        raise


def download_extractor(extractor_name):
    store = ExtractorMetadataStore()
    console.print("[bold #4AA4F4]Downloading Extractor...[/]")
    extractors_index = ExtractorIndex()

    extractor_info = extractors_index.metadata_by_name(extractor_name)
    extractor_module_name = extractor_info["module_name"]

    fs = fsspec.filesystem("s3", anon=True)
    extractor_path = (
        f's3://indexifyextractors/indexify-extractors/{extractor_info["path"]}'
    )

    try:
        fs.get(extractor_path, EXTRACTOR_MODULE_PATH, recursive=True)
    except Exception as e:
        value = {
            "extractor_name": extractor_name,
            "stage": "extractor_download",
            "error": str(e),
        }
        log_event("extractor_download_failed", value=value)
        raise e
    base_extractor_path = os.path.basename(extractor_path)
    try:
        install_dependencies(os.path.join(EXTRACTOR_MODULE_PATH, base_extractor_path))
    except Exception as e:
        value = {
            "extractor_name": extractor_name,
            "stage": "install_dependencies",
            "error": str(e),
        }
        log_event("extractor_download_failed", value=value)
        raise e

    # Store the extractor info in the database

    try:
        description = ExtractorWrapper.from_name(
            f"indexify_extractors.{extractor_module_name}"
        ).describe()
    except Exception as e:
        value = {
            "extractor_name": extractor_name,
            "stage": "extractor_description",
            "error": str(e),
        }
        log_event("extractor_download_failed", value=value)
        raise e

    if description.name.startswith("tensorlake"):
        log_event("extractor_download", description.name)

    try:
        store.save_description(extractor_module_name, description)
    except Exception as e:
        print(f"Error saving extractor description: {e}")
        value = {
            "extractor_name": extractor_name,
            "stage": "save_description",
            "error": str(e),
        }
        log_event("extractor_download_failed", value=value)
        raise e

    # Print instruction last to improve user experience.
    print_instructions()
