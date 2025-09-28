import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

DEFAULT_MODEL_DIR = os.path.join(".", "models", "mistral")

class HFEngine:
    def __init__(self):
        model_dir = os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR)
        device = os.getenv("DEVICE", "cuda")

        logging.info(f"Loading model from {model_dir} on {device}")

        # Load from local folder
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
        self.streamer = None

    async def stream(self, chat_input, generation_parameters):
        from transformers import TextIteratorStreamer

        # Setup streamer
        self.streamer = TextIteratorStreamer(self.tokenizer)
        input_ids = self.tokenizer(chat_input, return_tensors="pt").input_ids.to(self.model.device)

        # Run generation in background thread
        generation_kwargs = dict(input_ids=input_ids, streamer=self.streamer, **generation_parameters)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Yield tokens as they stream
        for token in self.streamer:
            yield {"status": 200, "delta": token}
