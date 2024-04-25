from typing import List, Union
from indexify_extractor_sdk import Content, Extractor, Feature
from pydantic import BaseModel, Field
from transformers import pipeline
from openai import OpenAI

class SchemaExtractorConfig(BaseModel):
    service: str = Field(default='openai')
    key: str = Field(default=None)
    schema: dict = Field(default={'properties': {'name': {'title': 'Name', 'type': 'string'}}, 'required': ['name'], 'title': 'User', 'type': 'object'})
    data: str = Field(default=None)
    additional_messages: str = Field(default='Extract information in JSON according to this schema and return only the output. ')

class SchemaExtractor(Extractor):
    name = "tensorlake/schema"
    description = "An extractor that let's you extract JSON from schemas."
    system_dependencies = []
    input_mime_types = ["text/plain"]

    def __init__(self):
        super(SchemaExtractor, self).__init__()

    def extract(self, content: Content, params: SchemaExtractorConfig) -> List[Union[Feature, Content]]:
        contents = []
        text = content.data.decode("utf-8")

        service = params.service
        key = params.key
        schema = params.schema
        data = params.data
        additional_messages = params.additional_messages
        if data is None:
            data = text

        if service == "openai":
            if key is None:
                client = OpenAI()
            else:
                client = OpenAI(api_key=key)
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": additional_messages + str(schema)},
                    {"role": "user", "content": data}
                ]
            )
            response_content = response.choices[0].message.content
            feature = Feature.metadata(value={"model": response.model, "completion_tokens": response.usage.completion_tokens, "prompt_tokens": response.usage.prompt_tokens}, name="text")
        
        contents.append(Content.from_text(response_content, features=[feature]))
        
        return contents

    def sample_input(self) -> Content:
        return Content.from_text("Hello, I am Diptanu from Tensorlake.")

if __name__ == "__main__":
    schema = {'properties': {'invoice_number': {'title': 'Invoice Number', 'type': 'string'}, 'date': {'title': 'Date', 'type': 'string'}, 'account_number': {'title': 'Account Number', 'type': 'string'}, 'owner': {'title': 'Owner', 'type': 'string'}, 'address': {'title': 'Address', 'type': 'string'}, 'last_month_balance': {'title': 'Last Month Balance', 'type': 'string'}, 'current_amount_due': {'title': 'Current Amount Due', 'type': 'string'}, 'registration_key': {'title': 'Registration Key', 'type': 'string'}, 'due_date': {'title': 'Due Date', 'type': 'string'}}, 'required': ['invoice_number', 'date', 'account_number', 'owner', 'address', 'last_month_balance', 'current_amount_due', 'registration_key', 'due_date'], 'title': 'User', 'type': 'object'}
    data = Content.from_text('Axis\nSTATEMENTInvoice No. 20240501-336593\nDate: 4/19/2024\nAccount Number:\nOwner:\nProperty:922000203826\nJohn Doe\n200 Park Avenue, Manhattan\nJohn Doe\n200 Park Avenue Manhattan\nNew York 10166SUMMARY OF ACCOUNT\nLast Month Balance:\nCurrent Amount Due:$653.03\n$653.03\nAccount details on back.\nProfessionally\nprepared by:\nSTATEMENT MESSAGE\nWelcome to Action Property Management! We are excited to be\nserving your community. Our Community Care team is more than\nhappy to assist you with any billing questions you may have. For\ncontact options, please visit www.actionlife.com/contact. Visit the\nAction Property Management web page at: www.actionlife.com.BILLING QUESTIONS\nScan the QR code to\ncontact our\nCommunity Care\nteam.\nactionlife.com/contact\nCommunityCare@actionlife.com\nRegister your Resident\nPortal account now!\nRegistration Key/ID:\nFLOWR2U\nresident.actionlife.com\nTo learn more about issues facing HOAs, say "Hey Siri, search the web for The Uncommon Area by Action Property Management."\nMake checks payable to:\nAxisAccount Number: 922000203826\nOwner: John Doe\nPLEASE REMIT PAYMENT TO:\n** AUTOPAY SCHEDULED **\n** NO REMITTANCE NECESSARY **CURRENT AMOUNT DUE\n$653.03\nDUE DATE\n5/1/2024\n0049 00008330 0000922000203826 7 00065303 00000000 9')
    input_params = SchemaExtractorConfig(service="openai", schema=schema)
    extractor = SchemaExtractor()
    results = extractor.extract(data, params=input_params)
    print(results)