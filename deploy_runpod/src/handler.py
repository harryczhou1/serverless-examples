import os
import asyncio
from engine import HFEngine

engine = HFEngine()

async def handler():
    chat_input = [{"role": "user", "content": "Hello, test streaming."}]
    async for token in engine.stream(chat_input, {"max_new_tokens": 100, "temperature": 0.7}):
        print(token, flush=True)

if __name__ == "__main__":
    asyncio.run(handler())
