import json

from graphgen.bases import BaseExtractor, BaseLLMWrapper, Chunk
from graphgen.templates import SCHEMA_GUIDED_EXTRACTION_PROMPT
from graphgen.utils import detect_main_language, logger


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
        self.required_keys = self.schema.get("required")
        if not self.required_keys:
            # If no required keys are specified, use all keys from the schema as default
            self.required_keys = list(self.schema.get("properties", {}).keys())

    def build_prompt(self, text: str) -> str:
        schema_explanation = ""
        for field, details in self.schema.get("properties", {}).items():
            description = details.get("description", "No description provided.")
            schema_explanation += f'- "{field}": {description}\n'

        lang = detect_main_language(text)

        prompt = SCHEMA_GUIDED_EXTRACTION_PROMPT[lang].format(
            field=self.schema.get("name", "the document"),
            schema_explanation=schema_explanation,
            examples="",
            text=text,
        )
        return prompt

    async def extract(self, chunk: Chunk) -> dict:
        text = chunk.content

        prompt = self.build_prompt(text)
        response = await self.llm_client.generate_answer(prompt)
        try:
            extracted_info = json.loads(response)
            # Ensure all required keys are present
            for key in self.required_keys:
                if key not in extracted_info:
                    extracted_info[key] = ""
            if any(extracted_info[key] == "" for key in self.required_keys):
                logger.debug("Missing required keys in extraction: %s", extracted_info)
                return {}
            logger.debug("Extracted info: %s", extracted_info)
            return extracted_info

        except json.JSONDecodeError:
            logger.error("Failed to parse extraction response: %s", response)
            return {}
