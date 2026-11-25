"""Drop ComfyUI's cached models / allocator blocks to keep VRAM residency low.

    python free_vram.py [port] [--now]

ComfyUI's /free only *queues* the unload: the flag is consumed when the next
prompt starts. So this sets the flag and, with --now, submits a 64x64 no-op
prompt to trigger it immediately (costs a few seconds).

Why bother: a long-lived ComfyUI keeps models resident. The same z-image job
measured 24.3 s with a stale 16.8 GB resident set vs 10.4 s after a clean
restart, and H3 could not run twice in a row without hitting OOM.
"""
import json
import subprocess
import sys
import time
import urllib.request

args = [a for a in sys.argv[1:] if not a.startswith("--")]
port = int(args[0]) if args else 8188
now = "--now" in sys.argv
SERVER = "http://127.0.0.1:%d" % port


def vram_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return -1


def post(path, payload):
    req = urllib.request.Request(SERVER + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


before = vram_mb()
post("/free", {"unload_models": True, "free_memory": True})
time.sleep(3)
print("free flag sent (VRAM %d MB -> %d MB)" % (before, vram_mb()))

if now:
    noop = {
        "1": {"class_type": "EmptyImage", "inputs": {"width": 64, "height": 64, "batch_size": 1, "color": 0}},
        "2": {"class_type": "PreviewImage", "inputs": {"images": ["1", 0]}},
    }
    pid = json.loads(post("/prompt", {"prompt": noop, "client_id": "free-vram"}).decode())["prompt_id"]
    for _ in range(120):
        with urllib.request.urlopen(SERVER + "/history/" + pid, timeout=10) as r:
            if pid in json.loads(r.read().decode()):
                break
        time.sleep(0.5)
    time.sleep(2)
    after = vram_mb()
    print("after no-op prompt: VRAM %d MB (freed %d MB)" % (after, before - after))
else:
    print("will take effect when the next prompt starts; pass --now to force it")
