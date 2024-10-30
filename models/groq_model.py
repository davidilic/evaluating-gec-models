import os, time
from models.gec_model import GECModel
from groq import Groq
from dotenv import load_dotenv


class GroqGECModel(GECModel):

    def __init__(self, model_name: str, retries: int = 20, delay: float = 15.0):
        self.model_name = model_name
        self.retries = retries
        self.delay = delay
        load_dotenv()
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def correct_errors(self, text: str, text_language: str) -> str:
        """Produce an error-corrected version of the input text using Groq API with retries."""

        system_prompt = (
            "You are a helpful grammar correction assistant. "
            "Your task is to correct any grammatical errors in the provided text while maintaining its original meaning. "
            "Only return the corrected text without any explanations."
        )

        user_prompt = (
            f"Please correct any grammatical errors in the following {text_language} text. "
            "Only return the corrected text without any explanations or additional comments: "
            f"{text}"
        )

        for attempt in range(self.retries):
            try:
                response = self.client.chat.completions.create(
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                    model=self.model_name,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt < self.retries - 1:
                    time.sleep(self.delay)
                else:
                    raise Exception(f"Error calling Groq API after {self.retries} attempts: {str(e)}")


class Gemma9bGEC(GroqGECModel):
    def __init__(self):
        super().__init__(model_name="gemma2-9b-it")


class Llama90bGEC(GroqGECModel):
    def __init__(self):
        super().__init__(model_name="llama-3.2-90b-vision-preview")


class Llama11bGEC(GroqGECModel):
    def __init__(self):
        super().__init__(model_name="llama-3.2-11b-vision-preview")


class Mixtral8x7bGEC(GroqGECModel):
    def __init__(self):
        super().__init__(model_name="mixtral-8x7b-32768")
