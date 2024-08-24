from typing import List, Optional, Union

from indexify.extractor_sdk import Content, Extractor, Feature
from indexify.extractors.pdf_parser import Page, PageFragmentType, PDFParser
from pydantic import BaseModel, Field


class PDFExtractorConfig(BaseModel):
    language: Optional[str] = "en"


class PDFExtractor(Extractor):
    name = "tensorlake/pdfextractor"
    description = "PDF Extractor for Texts, Images & Tables"
    system_dependencies = ["poppler-utils"]
    input_mime_types = ["application/pdf"]

    def __init__(self):
        super(PDFExtractor, self).__init__()

    def extract(
        self, content: Content, params: PDFExtractorConfig
    ) -> List[Union[Feature, Content]]:
        contents = []
        pdf_parser = PDFParser(data=content.data)
        pages: List[Page] = pdf_parser.parse()
        for page in pages:
            text = ""
            for fragment in page.fragments:
                if fragment.fragment_type == PageFragmentType.TEXT:
                    text += fragment.text
                if fragment.fragment_type == PageFragmentType.TABLE:
                    contents.append(
                        Content(content_type="text/html", data=fragment.table.data)
                    )
                if fragment.fragment_type == PageFragmentType.FIGURE:
                    contents.append(
                        Content(content_type="image/png", data=fragment.image.data)
                    )
            if text != "":
                contents.append(Content(content_type="text/plain", data=text))
        return contents

    def sample_input(self) -> Content:
        config = PDFExtractorConfig()
        return (self.sample_scientific_pdf(), config.model_dump_json())


if __name__ == "__main__":
    f = open("1706.03762v7.pdf", "rb")
    pdf_data = Content(content_type="application/pdf", data=f.read())
    extractor = PDFExtractor()
    params = PDFExtractorConfig()
    results = extractor.extract(pdf_data, params)
    print(results)
