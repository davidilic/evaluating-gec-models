from typing import Protocol


class GECModel(Protocol):

    def correct_errors(self, text: str, text_language: str) -> str:
        """Produce an error-corrected version of the input text"""
        ...
