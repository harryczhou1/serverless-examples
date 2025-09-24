import os
import logging
import asyncio
from typing import List, Dict, Any, Union
from threading import Thread
from queue import Empty
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from constants import DEFAULT_MODEL_DIR, DEFAULT_MODEL_NAME, DEFAULT_DEVICE

class HFEngine:
    def __init__(self) -> None:
        model_dir = os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR)
        model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
        device = os.getenv("DEVICE", DEFAULT_DEVICE)

        # prefer local /model if it exists, else Hugging Face hub
        model_path = model_dir if os.path.exists(model_dir) and os.listdir(model_dir) else model_name

        logging.info(f"📦 Loading model from {model_path} on {device}")
        self.model, self.tokenizer, self.streamer = self._initialize_llm(model_path, device)
        self.device = device

    def _initialize_llm(self, model_name_or_path: str, device: str):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map="auto",
                torch_dtype="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            streamer = TextIteratorStreamer(tokenizer)
        except Exception as error:
            logging.error("❌ Error initializing HuggingFace engine: %s", error)
            raise error

        return model, tokenizer, streamer

    async def stream(self, chat_input: Union[str, List[Dict[str, str]]], generation_parameters: Dict[str, Any]):
        try:
            async for output in self._stream(chat_input, generation_parameters):
                yield output
        except Exception as e:
            yield {"error": str(e)}

    async def _stream(self, chat_input: Union[str, List[Dict[str, str]]], generation_parameters: Dict[str, Any]):
        if isinstance(chat_input, str):
            chat_input = [{"role": "user", "content": chat_input}]

        input_ids = self.tokenizer.apply_chat_template(
            conversation=chat_input, tokenize=True, return_tensors="pt"
        ).to(self.device)

        generation_kwargs = dict(input_ids=input_ids, streamer=self.streamer, **generation_parameters)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for next_token in self.streamer:
            try:
                if next_token is not None:
                    yield {"status": 200, "delta": next_token}
            except Empty:
                await asyncio.sleep(0.001)
