import os
import logging
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from constants import DEFAULT_MODEL_DIR

class HFEngine:
    def __init__(self):
        raw_model_dir = os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR)
        model_dir = Path(raw_model_dir).expanduser().resolve()
        device = os.getenv("DEVICE", "cuda")

        logging.info(f"🚀 Loading model from {model_dir} on {device}")

        # Sanity check: make sure config.json exists
        config_file = model_dir / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"❌ No config.json found in {model_dir}. Did you run the download script?")

        # Hugging Face expects a string path
        model_path_str = str(model_dir)

        # Load tokenizer + model from local folder
        self.tokenizer = AutoTokenizer.from_pretrained(model_path_str, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path_str, local_files_only=True).to(device)
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
