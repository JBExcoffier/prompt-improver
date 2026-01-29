import openai
from models.languagemodels.base import LanguageModel


class OpenAImodel(LanguageModel):
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=api_key)

    def get_string_output(self, messages: list[dict]) -> str:
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
        )

        return completion.choices[0].message.content
