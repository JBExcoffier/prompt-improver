import openai
from models.languagemodels.base import LanguageModel


class VLLMmodel(LanguageModel):
    def __init__(
        self, model_name: str, api_key: str, base_url: str, max_tokens: int = 10
    ):
        """
        vLLM model accessed through the OpenAI client.
        """
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.max_tokens = max_tokens

    def get_string_output(self, messages: list[dict]) -> str:
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            max_tokens=self.max_tokens,
        )

        return completion.choices[0].message.content
