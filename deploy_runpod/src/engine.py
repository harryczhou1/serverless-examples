import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

class HFEngine:
    def __init__(self):
        model_dir = "/model"  # always load from local /model
        device = os.getenv("DEVICE", "cuda")

        logging.info(f"🚀 Loading model from {model_dir} on {device}")

        # Load tokenizer + model from local files only
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True).to(device)
        self.streamer = TextIteratorStreamer(self.tokenizer)

    async def stream(self, chat_input, generation_parameters):
        """Stream output tokens back as they are generated."""
        input_ids = self.tokenizer.apply_chat_template(
            conversation=chat_input, tokenize=True, return_tensors="pt"
        ).to(self.model.device)

        generation_kwargs = dict(
            input_ids=input_ids,
            streamer=self.streamer,
            **generation_parameters
        )

        from threading import Thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for token in self.streamer:
            yield {"status": 200, "delta": token}
