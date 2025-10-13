from graphgen.bases import BaseGenerator


class MultiHopGenerator(BaseGenerator):
    def build_prompt(self, batch) -> str:
        pass

    def parse_response(self, response: str):
        pass
