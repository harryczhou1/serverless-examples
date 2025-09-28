import runpod
from engine import HFEngine

engine = HFEngine()

async def handler(event):
    user_input = event.get("input", {}).get("text", "")
    params = event.get("input", {}).get("params", {})
    default_params = {
        "temperature": 0.7,
        "top_p": 0.95,
        "do_sample": True,
        "max_new_tokens": 128,
    }
    default_params.update(params)

    output = []
    async for chunk in engine.stream(user_input, default_params):
        if "delta" in chunk:
            output.append(chunk["delta"])

    return {"output": "".join(output)}

runpod.serverless.start({"handler": handler})
