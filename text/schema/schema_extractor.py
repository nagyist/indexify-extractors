import base64
import io
import mimetypes
import os
from typing import Any, Dict, List, Optional, Union

from indexify_extractor_sdk import Content, Extractor, Feature
from openai import OpenAI
from pdf2image import convert_from_bytes
from pydantic import BaseModel, Field


class SchemaExtractorConfig(BaseModel):
    model: Optional[str] = Field(default="gpt-4o-2024-08-06")
    api_key: Optional[str] = Field(default=None)
    system_prompt: str = Field(default="Extract the information.")
    user_prompt: Optional[str] = Field(default=None)
    response_format: Optional[Dict[str, Any]] = Field(
        default={
            "properties": {"name": {"title": "Name", "type": "string"}},
            "required": ["name"],
            "title": "Event",
            "type": "object",
            "additionalProperties": False,
        }
    )


class SchemaExtractor(Extractor):
    name = "tensorlake/schema"
    description = "An extractor that let's you extract JSON from schemas."
    system_dependencies = []
    input_mime_types = [
        "text/plain",
        "application/json",
        "application/pdf",
        "image/jpeg",
        "image/png",
    ]

    def __init__(self):
        super(SchemaExtractor, self).__init__()

    def extract(
        self, content: Content, params: SchemaExtractorConfig
    ) -> List[Union[Feature, Content]]:
        contents = []
        model_name = params.model
        key = params.api_key
        prompt = params.system_prompt
        query = params.user_prompt
        response_format = params.response_format

        if content.content_type == "application/pdf":
            images = convert_from_bytes(content.data)

            all_responses = []
            for image in images:
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format="JPEG")
                img_byte_arr = img_byte_arr.getvalue()
                response = self._process_image(
                    img_byte_arr, model_name, key, prompt, query, response_format
                )
                all_responses.append(response)

            response_content = "\n\n".join(all_responses)

        elif content.content_type in ["image/jpeg", "image/png"]:
            response_content = self._process_image(
                content.data, model_name, key, prompt, query, response_format
            )

        else:
            text = content.data.decode("utf-8")
            if query is None:
                query = text
            response_content = self._process_text(
                model_name, key, prompt, query, response_format
            )

        contents.append(Content.from_text(response_content))
        return contents

    def _process_image(
        self, image_data, model_name, key, prompt, query, response_format
    ):
        client = self._get_client(key)

        encoded_image = base64.b64encode(image_data).decode("utf-8")

        messages_content = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    },
                ],
            },
        ]

        try:
            response = client.beta.chat.completions.parse(
                model=model_name,
                messages=messages_content,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "schema_response",  # Add a required name
                        "strict": True,
                        "schema": response_format,
                    },
                },
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Unable to process image: {str(e)}")
            raise e

    def _process_text(self, model_name, key, prompt, query, response_format):
        client = self._get_client(key)

        messages_content = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ]

        try:
            response = client.beta.chat.completions.parse(
                model=model_name,
                messages=messages_content,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "schema_response",  # Add a required name
                        "strict": True,
                        "schema": response_format,
                    },
                },
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Unable to process text: {str(e)}")
            raise e

    def _get_client(self, key):
        if ("OPENAI_API_KEY" not in os.environ) and (key is None):
            raise ValueError(
                "The OPENAI_API_KEY environment variable is not present and no API key was provided."
            )
        if ("OPENAI_API_KEY" in os.environ) and (key is None):
            return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        return OpenAI(api_key=key)

    def sample_input(self) -> Content:
        return Content.from_text("Alice and Bob are going to a science fair on Friday.")


if __name__ == "__main__":
    json_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "date": {"type": "string"},
            "participants": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["name", "date", "participants"],
        "additionalProperties": False,
    }

    prompt = "Extract the event information."
    article = Content.from_text("Alice and Bob are going to a science fair on Friday.")
    input_params = SchemaExtractorConfig(
        model="gpt-4o-2024-08-06",
        system_prompt=prompt,
        response_format=json_schema,  # Use the JSON schema here
    )
    extractor = SchemaExtractor()
    results = extractor.extract(article, params=input_params)
    print(results)
