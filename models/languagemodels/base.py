import abc


class LanguageModel(abc.ABC):
    @abc.abstractmethod
    def get_string_output(self, messages: list[dict]) -> str:
        """
        Method must be implemented by inherited classes.
        """
        pass
