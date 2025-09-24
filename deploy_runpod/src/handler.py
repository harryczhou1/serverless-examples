import runpod
import asyncio
from engine import HFEngine

engine = HFEngine()

async def handler(event):
    """
    RunPod Serverless handler.
    Expects input:
    {
      "input": {
        "text": "hello",
        "params": {"max_new_tokens": 100}
      }
    }
    """
    user_input = event.get("input", {}).get("text", "")
    params = event.get("input", {}).get("params", {})

    default_params = {
        "temperature": 0.7,
        "top_p": 0.95,
        "do_sample": True,
        "max_new_tokens": 128,
    }
    default_params.update(params)

    output_tokens = []
    async for chunk in engine.stream(user_input, default_params):
        if "delta" in chunk:
            output_tokens.append(chunk["delta"])

    return {"output": "".join(output_tokens)}

runpod.serverless.start({"handler": handler})
