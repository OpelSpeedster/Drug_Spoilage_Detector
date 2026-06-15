"""Modal inference backend for Drug Spoilage Detector.

Deploys MiniCPM-V 2.6 INT4 (8B params, bitsandbytes quantized) via vLLM
with an HTTPS endpoint. The Gradio frontend calls this endpoint for VLM analysis.

Usage:
    modal deploy backend.py

This creates a persistent HTTPS endpoint. Set the URL as a HF Spaces secret
named MODAL_ENDPOINT_URL.
"""

import modal

MODEL_NAME = "openbmb/MiniCPM-V-2_6-int4"
VLLM_PORT = 8000

app = modal.App("biochem-spoilage-detect")

hf_cache = modal.Volume.from_name("biochem-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("biochem-vllm-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "vllm==0.6.6",
        "transformers==4.45.2",
        "fastapi",
        "bitsandbytes>=0.43.1",
        "sentencepiece==0.1.99",
        "accelerate==0.30.1",
        "pillow>=10.0.0",
    )
    .apt_install("git")
)


@app.cls(
    image=image,
    gpu="L4",
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    secrets=[modal.Secret.from_name("huggingface")],
    scaledown_window=5 * 60,
    timeout=15 * 60,
)
class VLMInference:
    @modal.enter()
    def start(self):
        import subprocess, time, os

        hf_token = os.environ.get("TOKEN", "")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

        cmd = [
            "vllm", "serve",
            MODEL_NAME,
            "--port", str(VLLM_PORT),
            "--dtype", "auto",
            "--max-model-len", "8192",
            "--gpu-memory-utilization", "0.9",
            "--trust-remote-code",
            "--host", "0.0.0.0",
            "--quantization", "bitsandbytes",
            "--load-format", "bitsandbytes",
            "--limit-mm-per-prompt", "image=8",
        ]

        print("Starting vLLM:", " ".join(cmd))
        self.proc = subprocess.Popen(cmd)

        deadline = time.time() + 600
        while time.time() < deadline:
            try:
                import urllib.request
                urllib.request.urlopen(f"http://localhost:{VLLM_PORT}/health", timeout=5)
                print("vLLM ready")
                break
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError("vLLM failed to start within 10 min")

    @modal.fastapi_endpoint(method="POST")
    def analyze(self, images: list[str], prompt: str) -> dict:
        import json, urllib.request

        content = [{"type": "text", "text": prompt}]
        for img_b64 in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            })

        messages = [{"role": "user", "content": content}]

        payload = json.dumps({
            "model": MODEL_NAME,
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.1,
        }).encode()

        req = urllib.request.Request(
            f"http://localhost:{VLLM_PORT}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read())

        text = result["choices"][0]["message"]["content"]
        return {"response": text}
