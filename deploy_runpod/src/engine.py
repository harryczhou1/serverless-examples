import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from app_constants import DEFAULT_MODEL_DIR  # or constants

class HFEngine:
    def __init__(self):
        model_dir = os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR)
        device = os.getenv("DEVICE", "cuda")

        logging.info(f"Loading model from {model_dir} on {device}")

        # Load from local model folder
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
        self.streamer = TextIteratorStreamer(self.tokenizer)

    async def stream(self, chat_input, generation_parameters):
        # existing streaming code …
        input_ids = self.tokenizer.apply_chat_template(
            conversation=chat_input, tokenize=True, return_tensors="pt"
        ).to(self.model.device)
        generation_kwargs = dict(input_ids=input_ids, streamer=self.streamer, **generation_parameters)
        from threading import Thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        for token in self.streamer:
            yield {"status": 200, "delta": token}
