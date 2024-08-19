import json
from typing import Dict
from dataclasses import dataclass
from urllib.parse import urlparse

import boto3
import httpx
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from google.cloud import storage
from google.protobuf.json_format import MessageToDict
from indexify.extractor_sdk import Content

from . import coordinator_service_pb2
from .base_extractor import ExtractorPayload


@dataclass
class UrlConfig:
    url: str
    config: Dict[str, str]


def disk_loader(file_path: str):
    file_path = file_path.removeprefix("file:/")
    with open(file_path, "rb") as f:
        return f.read()


def s3_loader(s3_url: str) -> bytes:
    parsed_url = urlparse(s3_url)
    bucket_name = parsed_url.netloc
    key = parsed_url.path.lstrip("/")

    s3 = boto3.client("s3")

    response = s3.get_object(Bucket=bucket_name, Key=key)
    return response["Body"].read()


def azure_blob_loader(blob_url: str) -> bytes:
    token_credential = DefaultAzureCredential()
    parsed_url = urlparse(blob_url)
    account_url = f"https://{parsed_url.netloc}"
    container_name = parsed_url.path.split("/")[1]
    blob_name = "/".join(parsed_url.path.split("/")[2:])

    blob_service_client = BlobServiceClient(
        account_url=account_url, credential=token_credential
    )
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=blob_name
    )

    return blob_client.download_blob().readall()


def gcp_storage_loader(storage_url: str) -> bytes:
    parsed_url = urlparse(storage_url)
    bucket_name = parsed_url.netloc
    blob_name = parsed_url.path.lstrip("/")

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    blob = bucket.blob(blob_name)
    return blob.download_as_bytes()


async def fetch_url(url_config: UrlConfig) -> bytes:
    try:
        kwargs = {}
        if url_config.config.get("use_tls"):
            kwargs["cert"] = (
                url_config.config["tls_config"]["cert_path"],
                url_config.config["tls_config"]["key_path"],
            )
            kwargs["verify"] = url_config.config["tls_config"]["ca_bundle_path"]
            kwargs["http2"] = True

        async with httpx.AsyncClient(**kwargs) as client:
            print(f"downloading url {url_config.url}")
            response = await client.get(url_config.url, follow_redirects=True)
            response.raise_for_status()
            return response.read()
    except Exception as e:
        raise e


async def download_content(
    task: coordinator_service_pb2.Task,
    url_config: UrlConfig,
) -> tuple[str, ExtractorPayload]:
    if url_config.url.startswith("file://"):
        input_bytes = disk_loader(url_config.url)
    elif url_config.url.startswith("s3://"):
        input_bytes = s3_loader(url_config.url)
    elif url_config.url.startswith("https://") or url_config.url.startswith("http://"):
        input_bytes = await fetch_url(url_config)
    elif url_config.url.startswith("gs://"):
        input_bytes = gcp_storage_loader(url_config.url)
    else:
        raise Exception(f"unsupported storage url {url_config.url}")

    extract_args = json.loads(task.input_params) if task.input_params else None

    return (
        task.id,
        ExtractorPayload(
            data=input_bytes,
            content_type=task.content_metadata.mime,
            extract_args=extract_args,
        ),
    )
