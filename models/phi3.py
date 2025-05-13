from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class Phi3Model:
    def __init__(self):
        self.model_id = "microsoft/phi-3-mini-4k-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto"
        )
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1  # CPU only
        )

    def generate(self, prompt: str) -> str:
        output = self.pipeline(prompt, max_new_tokens=200, do_sample=True)
        return output[0]["generated_text"]