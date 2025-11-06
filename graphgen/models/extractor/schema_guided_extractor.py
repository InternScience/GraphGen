from graphgen.bases import BaseExtractor, BaseLLMWrapper


class SchemaGuidedExtractor(BaseExtractor):
    """
    Use JSON/YAML Schema or Pydantic Model to guide the LLM to extract structured information from text.

    Usage example:
        schema = {
                "type": "legal contract",
                "description": "A legal contract for leasing property.",
                "properties": {
                    "end_date": {"type": "string", "description": "The end date of the lease."},
                    "leased_space": {"type": "string", "description": "Description of the space that is being leased."},
                    "lessee": {"type": "string", "description": "The lessee's name (and possibly address)."},
                    "lessor": {"type": "string", "description": "The lessor's name (and possibly address)."},
                    "signing_date": {"type": "string", "description": "The date the contract was signed."},
                    "start_date": {"type": "string", "description": "The start date of the lease."},
                    "term_of_payment": {"type": "string", "description": "Description of the payment terms."},
                    "designated_use": {"type": "string",
                    "description": "Description of the designated use of the property being leased."},
                    "extension_period": {"type": "string",
                    "description": "Description of the extension options for the lease."},
                    "expiration_date_of_lease": {"type": "string", "description": "The expiration data of the lease."}
                },
                "required": ["lessee", "lessor", "start_date", "end_date"]
            }
        extractor = SchemaGuidedExtractor(llm_client, schema)
        result = extractor.extract(text)

    """

    def __init__(self, llm_client: BaseLLMWrapper, schema: dict):
        super().__init__(llm_client)
        self.schema = schema

    def build_prompt(self, text: str) -> str:
        pass

    def extract(self, text_or_documents: str) -> dict:
        pass
