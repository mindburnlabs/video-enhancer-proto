from fastapi import FastAPI, Response
import os
from prometheus_client import CollectorRegistry, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI()

@app.get("/health")
def health():
    try:
        import torch
        gpu_ok = torch.cuda.is_available()
        vram = torch.cuda.get_device_properties(0).total_memory if gpu_ok else 0
    except ImportError:
        gpu_ok = False
        vram = 0
    
    return {
        "status": "ok",
        "gpu": gpu_ok,
        "vram_bytes": int(vram),
        "env": list(sorted(os.environ.keys()))[:5]
    }

_registry = CollectorRegistry()
latency_g = Gauge("ve_latency_ms_per_frame", "Latency by strategy", ["strategy"], registry=_registry)
vram_g = Gauge("ve_vram_bytes", "VRAM peak", ["strategy"], registry=_registry)

@app.get("/metrics")
def metrics():
    return Response(generate_latest(_registry), media_type=CONTENT_TYPE_LATEST)