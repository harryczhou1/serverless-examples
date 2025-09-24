import os
import logging
from threading import Thread
from queue import Empty
import asyncio
from typing import List, Dict, Any, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from constants import DEFAULT_DEVICE, DEFAULT_MODEL_DIR

class HFEngine:
    def __init__(self):
        self.device = os.getenv("DEVICE", DEFAULT_DEVICE)
        model_dir = os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR)

        # Initialize model/tokenizer with accelerate
        self.model, self.tokenizer, self.streamer = self._initialize_llm(model_dir, self.device)

    def _initialize_llm(self, model_dir: str, device: str):
        try:
            print(f"📦 Loading model from {model_dir} on {device} ...")
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                device_map="auto",
                load_in_8bit=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            streamer = TextIteratorStreamer(tokenizer)
        except Exception as error:
            logging.error("Error initializing HuggingFace engine: %s", error)
            raise error
        return model, tokenizer, streamer

    async def stream(self, chat_input: Union[str, List[Dict[str, str]]], generation_parameters: Dict[str, Any]):
        if isinstance(chat_input, str):
            chat_input = [{"role": "user", "content": chat_input}]

        input_ids = self.tokenizer.apply_chat_template(conversation=chat_input, tokenize=True, return_tensors="pt").to(self.device)
        generation_kwargs = dict(input_ids=input_ids, streamer=self.streamer, **generation_parameters)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for next_token in self.streamer:
            try:
                if next_token is not None:
                    yield {"status": 200, "delta": next_token}
            except Empty:
                await asyncio.sleep(0.001)
