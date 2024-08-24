import io
import json
from typing import List

from faster_whisper import WhisperModel
from indexify_extractor_sdk import Content, Extractor
from pydantic import BaseModel


class InputParams(BaseModel):
    model: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"


class FasterWhisper(Extractor):
    name = "tensorlake/fasterwhisper"
    description = "fasterwhisper transcripts audio into json with timestamps and text"
    system_dependencies = []
    input_mime_types = ["audio", "audio/mpeg"]

    def __init__(self):
        super().__init__()

    def extract(self, content: Content, params: InputParams) -> List[Content]:
        # Wrap the content data in io.BytesIO
        audio_stream = io.BytesIO(content.data)

        model = WhisperModel(
            params.model, device=params.device, compute_type=params.compute_type
        )

        segments, info = model.transcribe(audio_stream, beam_size=5)

        entries = []
        for segment in segments:
            entries.append(
                {
                    "timestamp": {"start": segment.start, "end": segment.end},
                    "text": segment.text.strip(),
                }
            )

        return [
            Content.from_json(entries),
        ]

    def sample_input(self) -> Content:
        return self.sample_mp3()


if __name__ == "__main__":
    print(FasterWhisper().sample_input().data)
