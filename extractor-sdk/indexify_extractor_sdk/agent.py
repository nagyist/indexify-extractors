import asyncio
import ssl
import json
import grpc
import yaml
from typing import List, Dict, Union, Optional
from concurrent.futures.process import BrokenProcessPool

from pydantic import BaseModel, Json
import websockets
from websockets.exceptions import ConnectionClosed
from indexify.extractor_sdk import Content, Feature

from . import coordinator_service_pb2
from .coordinator_service_pb2_grpc import CoordinatorServiceStub
from .content_downloader import (
    download_content,
    UrlConfig,
)
from .metadata_store import ExtractorMetadataStore
from .extractor_worker import ExtractorWorker
from .base_extractor import ExtractorPayload
from .ingestion_api_models import (
    ApiContent,
    ApiFeature,
    BeginExtractedContentIngest,
    ExtractedFeatures,
    FinishExtractedContentIngest,
    ApiBeginExtractedContentIngest,
    ApiExtractedFeatures,
    ApiFinishExtractedContentIngest,
    ApiBeginMultipartContent,
    BeginMultipartContent,
    ApiFinishMultipartContent,
    ApiMultipartContentFrame,
    MultipartContentFrame,
    FinishMultipartContent,
)
from .server import http_server, ServerRouter, get_server_advertise_addr
from .task_store import TaskStore, CompletedTask

CONTENT_FRAME_SIZE = 1024 * 1024

MAX_MESSAGE_LENGTH = 16 * 1024 * 1024

# maximum number of content in extractor call
DEFAULT_BATCH_SIZE = 10


def begin_message(task_outcome, task: coordinator_service_pb2.Task, _executor_id):
    return ApiBeginExtractedContentIngest(
        BeginExtractedContentIngest=BeginExtractedContentIngest(
            task_id=task_outcome.task_id,
            executor_id=_executor_id,
            task_outcome=task_outcome.task_outcome,
        )
    )


async def send_extracted_content(ws, content: ApiContent, id: int, frame_size):
    # start new multipart content
    await ws.send(
        ApiBeginMultipartContent(
            BeginMultipartContent=BeginMultipartContent(id=id)
        ).model_dump_json()
    )

    # send data in chunks of frame_size
    for i in range(0, len(content.bytes), frame_size):
        slice = content.bytes[i : i + frame_size]
        content_frame = ApiMultipartContentFrame(
            MultipartContentFrame=MultipartContentFrame(bytes=slice)
        )
        await ws.send(content_frame.model_dump_json())

    # finish multipart content with features
    await ws.send(
        ApiFinishMultipartContent(
            FinishMultipartContent=FinishMultipartContent(
                content_type=content.content_type,
                features=content.features,
                labels=content.labels,
            )
        ).model_dump_json()
    )


class TaskReportError(Exception):
    """Exception raised for errors in the task reporting process."""

    def __init__(self, task_id, message="Failed to report task"):
        self.task_id = task_id
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"


async def process_task_outcome(
    task_outcome: CompletedTask,
    task: coordinator_service_pb2.Task,
    url,
    _executor_id,
    ssl_context,
    frame_size=CONTENT_FRAME_SIZE,
):
    try:
        async with websockets.connect(
            url, ssl=ssl_context, ping_interval=5, ping_timeout=30
        ) as ws:
            # start new extracted content ingest
            await ws.send(
                begin_message(task_outcome, task, _executor_id).model_dump_json()
            )

            num_extracted_content = 0
            if task_outcome.task_outcome == "Success":
                num_extracted_content = len(task_outcome.new_content)
                # send all contents one at a time
                for i, content in enumerate(task_outcome.new_content):
                    await send_extracted_content(
                        ws, content, id=i + 1, frame_size=frame_size
                    )

                # send all features one at a time
                for feature in task_outcome.features:
                    extracted_features = ApiExtractedFeatures(
                        ExtractedFeatures=ExtractedFeatures(
                            content_id=task.content_metadata.id, features=[feature]
                        )
                    )
                    await ws.send(extracted_features.model_dump_json())
                print(
                    f"finished message {FinishExtractedContentIngest(num_extracted_content=len(task_outcome.new_content)).model_dump_json()}"
                )

            # finish extracted content ingest
            finish_msg = ApiFinishExtractedContentIngest(
                FinishExtractedContentIngest=FinishExtractedContentIngest(
                    num_extracted_content=num_extracted_content
                )
            )

            await ws.send(finish_msg.model_dump_json())

            response = await ws.recv()
            response_data = json.loads(response)
            if "Error" in response_data:
                raise TaskReportError(task_outcome.task_id, response_data["Error"])

    except ConnectionClosed as e:
        if not e.rcvd is None:
            # the connection was closed by the server with an error message
            raise TaskReportError(
                task_outcome.task_id,
                f"Connection closed with code {e.code} reason {e.reason}",
            )
        else:
            # otherwise abnormal close, retry
            raise e


class ContentBatch(BaseModel):
    extractor: str
    content_list: Dict[str, ExtractorPayload] = {}


class ExtractorState(BaseModel):
    pending_batches: int = 0
    new_batches: List[ContentBatch]


def extractor_state_new(extractor: str) -> ExtractorState:
    return ExtractorState(new_batches=[ContentBatch(extractor=extractor)])


class ExtractTask(asyncio.Task):
    def __init__(
        self,
        *,
        extractor_worker: ExtractorWorker,
        extractor_module_class: str,
        extractor_name: str,
        content_batch: ContentBatch,
        **kwargs,
    ):
        kwargs["name"] = "extract_content"
        kwargs["loop"] = asyncio.get_event_loop()
        super().__init__(
            extractor_worker.async_submit(
                extractor=extractor_name,
                extractor_module_class=f"indexify_extractors.{extractor_module_class}",
                inputs=content_batch.content_list,
            ),
            **kwargs,
        )
        self.extractor_name = extractor_name
        self.task_ids = list(content_batch.content_list.keys())


class DownloadContentTask(asyncio.Task):
    def __init__(
        self,
        *,
        task: coordinator_service_pb2.Task,
        url_config: UrlConfig,
        **kwargs,
    ):
        kwargs["name"] = "download_content"
        kwargs["loop"] = asyncio.get_event_loop()
        super().__init__(
            download_content(task, url_config),
            **kwargs,
        )
        self.task_id = task.id


class ExtractorAgent:
    def __init__(
        self,
        executor_id: str,
        metadata_store: ExtractorMetadataStore,
        extractors: List[coordinator_service_pb2.Extractor],
        extractor_worker: ExtractorWorker,
        coordinator_addr: str,
        num_workers,
        listen_port: int,
        advertise_addr: Optional[str],
        ingestion_addr: str = "localhost:8900",
        config_path: Optional[str] = None,
        download_method: str = "direct",
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        self.num_workers = num_workers
        self._use_tls = False
        if config_path:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                self._config = config
            if config.get("use_tls", False):
                print("Running the extractor with TLS enabled")
                self._use_tls = True
                tls_config = config["tls_config"]
                self._ssl_context = ssl.create_default_context(
                    ssl.Purpose.SERVER_AUTH, cafile=tls_config["ca_bundle_path"]
                )
                self._ssl_context.load_cert_chain(
                    certfile=tls_config["cert_path"], keyfile=tls_config["key_path"]
                )
                self._protocol = "wss"
                self._tls_config = tls_config
            else:
                self._ssl_context = None
                self._protocol = "ws"
        else:
            self._ssl_context = None
            self._protocol = "ws"
            self._config = {}

        self._task_store: TaskStore = TaskStore()
        self._executor_id = executor_id
        self._metadata_store = metadata_store
        self._extractors = extractors
        self._has_registered = False
        self._coordinator_addr = coordinator_addr
        self._ingestion_addr = ingestion_addr
        self._listen_port = listen_port
        self._advertise_addr = advertise_addr
        self._download_method = download_method
        self._batch_size = batch_size
        self._extractor_worker = extractor_worker

    async def ticker(self):
        while True:
            await asyncio.sleep(5)
            yield coordinator_service_pb2.HeartbeatRequest(
                executor_id=self._executor_id,
                pending_tasks=self._task_store.num_pending_tasks(),
                max_pending_tasks=self._batch_size * 2,
            )

    async def register(self):
        # This needs to be here because every time we register we need to create a new channel
        # because the old one might have been closed after hb was broken or we could never connect
        if self._use_tls:
            # Load the certificate and key files
            with open(self._config["tls_config"]["cert_path"], "rb") as f:
                cert = f.read()
            with open(self._config["tls_config"]["key_path"], "rb") as f:
                key = f.read()
            with open(self._config["tls_config"]["ca_bundle_path"], "rb") as f:
                ca_cert = f.read()
            credentials = grpc.ssl_channel_credentials(
                root_certificates=ca_cert, private_key=key, certificate_chain=cert
            )
            self._channel = grpc.aio.secure_channel(
                self._coordinator_addr,
                credentials=credentials,
                options=[
                    ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
                    ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
                ],
            )
        else:
            self._channel = grpc.aio.insecure_channel(
                self._coordinator_addr,
                options=[
                    ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
                    ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
                ],
            )

        self._stub: CoordinatorServiceStub = CoordinatorServiceStub(self._channel)
        req = coordinator_service_pb2.RegisterExecutorRequest(
            executor_id=self._executor_id,
            addr=self._advertise_addr,
            extractors=self._extractors,
        )
        return await self._stub.RegisterExecutor(req)

    async def task_completion_reporter(self):
        print("starting task completion reporter")
        # We should copy only the keys and not the values
        url = f"{self._protocol}://{self._ingestion_addr}/write_content"
        while True:
            outcomes = await self._task_store.task_outcomes()
            for task_outcome in outcomes:
                print(
                    f"reporting outcome of task {task_outcome.task_id}, outcome: {task_outcome.task_outcome}, num_content: {len(task_outcome.new_content)}, num_features: {len(task_outcome.features)}"
                )
                task: coordinator_service_pb2.Task = self._task_store.get_task(
                    task_outcome.task_id
                )
                try:
                    await process_task_outcome(
                        task_outcome, task, url, self._executor_id, self._ssl_context
                    )
                except TaskReportError as e:
                    print(f"failed to report task {e.task_id}, exception: {e}")
                    self._task_store.report_failed(task_id=e.task_id)
                    continue
                except Exception as e:
                    # the connection was dropped in the middle of the reporting process, retry
                    print(
                        f"failed to report task {task_outcome.task_id}, exception: {e}, retrying"
                    )
                    continue

                self._task_store.mark_reported(task_id=task_outcome.task_id)

    def get_url_config(self, task: coordinator_service_pb2.Task) -> UrlConfig:
        if self._download_method == "server-proxy":
            protocol = "https://" if self._config.get("use_tls") else "http://"
            url = f"{protocol}{self._ingestion_addr}/namespaces/{task.content_metadata.namespace}/content/{task.content_metadata.id}/download"
            return UrlConfig(url=url, config=self._config)
        else:
            return UrlConfig(url=task.content_metadata.storage_url, config={})

    async def task_launcher(self):
        async_tasks: List[asyncio.Task] = []
        extractor_states: Dict[str, ExtractorState] = {}
        async_tasks.append(
            asyncio.create_task(
                self._task_store.get_runnable_tasks(), name="get_runnable_tasks"
            )
        )
        while True:
            extractor: str
            state: ExtractorState
            for extractor, state in extractor_states.items():
                if (
                    state.pending_batches == 0
                    and len(state.new_batches[0].content_list) != 0
                ):
                    content_batch = state.new_batches.pop(0)
                    print(
                        f"extracting content for {extractor} tasks {len(content_batch.content_list.keys())} {content_batch.content_list.keys()}"
                    )
                    extractor_module_class = (
                        self._metadata_store.extractor_module_class(extractor)
                    )
                    async_tasks.append(
                        ExtractTask(
                            extractor_worker=self._extractor_worker,
                            extractor_module_class=extractor_module_class,
                            extractor_name=extractor,
                            content_batch=content_batch,
                        )
                    )
                    if len(state.new_batches) == 0:
                        state.new_batches.append(ContentBatch(extractor=extractor))
                    state.pending_batches += 1

            done, pending = await asyncio.wait(
                async_tasks, return_when=asyncio.FIRST_COMPLETED
            )
            async_tasks: List[asyncio.Task] = list(pending)
            for async_task in done:
                if async_task.get_name() == "get_runnable_tasks":
                    result: Dict[str, coordinator_service_pb2.Task] = await async_task
                    task: coordinator_service_pb2.Task
                    for _, task in result.items():
                        url_config: UrlConfig = self.get_url_config(task)
                        async_tasks.append(
                            DownloadContentTask(task=task, url_config=url_config)
                        )
                    async_tasks.append(
                        asyncio.create_task(
                            self._task_store.get_runnable_tasks(),
                            name="get_runnable_tasks",
                        )
                    )
                elif async_task.get_name() == "download_content":
                    if async_task.exception():
                        print(
                            f"failed to download content {async_task.exception()} for task {async_task.task_id}"
                        )
                        completed_task = CompletedTask(
                            task_id=async_task.task_id,
                            task_outcome="Failed",
                            new_content=[],
                            features=[],
                        )
                        self._task_store.complete(outcome=completed_task)
                        continue
                    # Process all completed downloads and accumulate them in batches
                    # without creating extraction tasks right away.
                    task_id: str
                    extractor_payload: ExtractorPayload
                    task_id, extractor_payload = await async_task
                    task: coordinator_service_pb2.Task = self._task_store.get_task(
                        task_id
                    )
                    state: ExtractorState = extractor_states.setdefault(
                        task.extractor, extractor_state_new(task.extractor)
                    )
                    if len(state.new_batches[-1].content_list) == self._batch_size:
                        state.new_batches.append(ContentBatch(extractor=task.extractor))
                    content_batch = state.new_batches[-1]
                    content_batch.content_list[task_id] = extractor_payload
                elif async_task.get_name() == "extract_content":
                    async_task: ExtractTask
                    state: ExtractorState = extractor_states[async_task.extractor_name]
                    state.pending_batches -= 1
                    try:
                        outputs = await async_task
                        task_id: str
                        e_output: List[Union[Feature, Content]]
                        for task_id, e_output in outputs.items():
                            print(f"completed task {task_id}")
                            new_content: List[ApiContent] = []
                            new_features: List[ApiFeature] = []
                            out: Union[Feature, Content]
                            for out in e_output:
                                if type(out) == Feature:
                                    new_features.append(
                                        ApiFeature.from_feature(feature=out)
                                    )
                                    continue
                                new_content.append(ApiContent.from_content(content=out))
                            completed_task = CompletedTask(
                                task_id=task_id,
                                task_outcome="Success",
                                new_content=new_content,
                                features=new_features,
                            )
                            self._task_store.complete(outcome=completed_task)
                    except BrokenProcessPool:
                        for task_id in async_task.task_ids:
                            self._task_store.retriable_failure(task_id)
                        continue
                    except Exception as e:
                        task_ids = ",".join(async_task.task_ids)
                        print(f"failed to execute tasks {task_ids} {e}")
                        for task_id in async_task.task_ids:
                            completed_task = CompletedTask(
                                task_id=task_id,
                                task_outcome="Failed",
                                new_content=[],
                                features=[],
                            )
                            self._task_store.complete(outcome=completed_task)
                        continue

    async def run(self):
        import signal

        asyncio.get_event_loop().add_signal_handler(
            signal.SIGINT, self.shutdown, asyncio.get_event_loop()
        )
        server_router = ServerRouter(self._extractor_worker)
        self._http_server = http_server(server_router, port=self._listen_port)
        asyncio.create_task(self._http_server.serve())
        if not self._advertise_addr:
            self._advertise_addr = await get_server_advertise_addr(self._http_server)
        print(f"advertise addr is {self._advertise_addr}")
        asyncio.create_task(self.task_launcher())
        asyncio.create_task(self.task_completion_reporter())
        self._should_run = True
        while self._should_run:
            print("attempting to register")
            try:
                await self.register()
                self._has_registered = True
            except Exception as e:
                print(f"failed to register: {e}")
                await asyncio.sleep(5)
                continue
            hb_ticker = self.ticker()
            print("starting heartbeat")
            try:
                hb_response_it = self._stub.Heartbeat(hb_ticker)
                resp: coordinator_service_pb2.HeartbeatResponse
                async for resp in hb_response_it:
                    self._task_store.add_tasks(resp.tasks)
            except Exception as e:
                print(f"failed to heartbeat{e}")
                continue

    async def _shutdown(self, loop):
        print("shutting down agent ...")
        self._should_run = False
        self._http_server.should_exit = True
        await self._channel.close()
        for task in asyncio.all_tasks(loop):
            task.cancel()

    def shutdown(self, loop):
        self._extractor_worker.shutdown()
        loop.create_task(self._shutdown(loop))
