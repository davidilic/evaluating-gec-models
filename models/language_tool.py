import language_tool_python
from models.gec_model import GECModel


class LanguageTool(GECModel):

    def __init__(self):
        self.english_tool = language_tool_python.LanguageTool("en-US")
        self.german_tool = language_tool_python.LanguageTool("de-DE")

    def correct_errors(self, text: str, text_language: str) -> str:
        if text_language == "english":
            return self.english_tool.correct(text)
        elif text_language == "german":
            return self.german_tool.correct(text)
        else:
            return text
