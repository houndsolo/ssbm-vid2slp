"""
check_env.py - confirm the GPU / PyTorch setup is what this project expects.

Checks:
  - torch is a cu126 build and older than 2.15 (newer wheels dropped Volta/V100 support)
  - the V100 is visible and is torch's default device "cuda:0"
  - an fp16 matmul runs and is accurate
  - fp16 autocast + GradScaler and scaled_dot_product_attention work

NOTE: this machine also has a small GeForce GT 1030. nvidia-smi lists the GT 1030 as
GPU 0 and the V100 as GPU 1, but PyTorch orders GPUs fastest-first, so in PyTorch the
V100 is cuda:0. If you ever set CUDA_VISIBLE_DEVICES, remember the two tools number
the cards differently. This script fails loudly if cuda:0 is not the V100.

Run:
    .venv/bin/python scripts/check_env.py
"""

import time

import torch

print("torch", torch.__version__, "| CUDA build", torch.version.cuda)
assert torch.version.cuda.startswith("12.6"), "expected a cu126 build of torch"
major, minor = (int(x) for x in torch.__version__.split(".")[:2])
assert (major, minor) < (2, 15), "torch >= 2.15 dropped sm_70 (V100) support"
assert "sm_70" in torch.cuda.get_arch_list(), "this torch build has no V100 (sm_70) kernels"

for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"cuda:{i} = {p.name} sm_{p.major}{p.minor} {p.total_memory / 2**30:.1f} GiB")
p = torch.cuda.get_device_properties(0)
assert (p.major, p.minor) == (7, 0) and p.total_memory > 30 * 2**30, "cuda:0 is not the 32 GB V100"

dev = torch.device("cuda:0")
a = torch.randn(8192, 8192, device=dev, dtype=torch.float16)
b = torch.randn(8192, 8192, device=dev, dtype=torch.float16)
ref = a.float() @ b.float()
c = a @ b
torch.cuda.synchronize()
t = time.time()
for _ in range(20):
    c = a @ b
torch.cuda.synchronize()
dt = (time.time() - t) / 20
err = ((c.float() - ref).abs().max() / ref.abs().max()).item()
print(f"fp16 matmul 8192^3: {dt * 1000:.1f} ms = {2 * 8192**3 / dt / 1e12:.0f} TFLOPS, max rel err {err:.1e}")
assert err < 1e-2

# fp16 autocast + GradScaler (no bf16 on V100)
m = torch.nn.Linear(512, 512).to(dev)
opt = torch.optim.SGD(m.parameters(), lr=0.01)
scaler = torch.amp.GradScaler("cuda")
with torch.autocast("cuda", dtype=torch.float16):
    y = m(torch.randn(64, 512, device=dev))
    loss = y.float().pow(2).mean()
scaler.scale(loss).backward()
scaler.step(opt)
scaler.update()
assert y.dtype == torch.float16
q = torch.randn(2, 8, 256, 64, device=dev, dtype=torch.float16)
torch.nn.functional.scaled_dot_product_attention(q, q, q)
print("autocast fp16 + GradScaler + SDPA: OK")
print("ALL CHECKS PASSED")
