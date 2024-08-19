import unittest
from unittest import IsolatedAsyncioTestCase

from indexify_extractor_sdk.base_extractor import Content, ExtractorPayload
from indexify_extractor_sdk.extractor_worker import ExtractorWorker


class TestExtractorWorker(IsolatedAsyncioTestCase):
    def __init__(self, *args, **kwargs):
        super(TestExtractorWorker, self).__init__(*args, **kwargs)

    async def test_extract(self):
        extractor_worker = ExtractorWorker()
        out = await extractor_worker.async_submit(
            "tensorlake/mock",
            "indexify_extractor_sdk.mock_extractor:MockExtractor",
            {
                "task1": ExtractorPayload(
                    data=b"hello world",
                    content_type="text/plain",
                    extract_args={"a": 1, "b": "foo"},
                )
            },
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out["task1"]), 2)


if __name__ == "__main__":
    unittest.main()
