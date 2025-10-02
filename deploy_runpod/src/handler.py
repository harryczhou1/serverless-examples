import runpod
from engine import HFEngine

engine = HFEngine()

async def handler(event):
    # Accept both "text" and "prompt" for user input
    input_data = event.get("input", {})
    user_input = input_data.get("text") or input_data.get("prompt") or ""

    # Params can come from input.params or top-level input
    params = input_data.get("params", {})
    if not params:  
        # fallback: copy generation args if they were sent directly in input
        params = {
            k: v for k, v in input_data.items()
            if k in ["max_new_tokens", "temperature", "top_p", "do_sample"]
        }

    # Default generation parameters
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
