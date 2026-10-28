#!/usr/bin/env bash
# install_nvenc.sh - add the NVIDIA video-encoder library (NVENC), so ffmpeg can compress the
# rendered videos on the V100's built-in encoder instead of on the CPU (x264).
#
# Why: measured 2026-10-28 during a render, x264 used ~40% of all 16 CPU cores, as much as
# Dolphin itself, while the GPUs were ~1% busy. ffmpeg already supports h264_nvenc but fails
# with "Cannot load libnvidia-encode.so.1": the server has the headless driver 580.178.04,
# which leaves this library out. The package libnvidia-encode-580-server adds just that
# library for the SAME driver version. It does not replace the driver or the kernel module,
# so no reboot should be needed and running jobs (CUDA, renders) keep working.
#
# Run it yourself (needs sudo):
#     bash scripts/install_nvenc.sh
set -euo pipefail

PKG=libnvidia-encode-580-server

# 1. The library must be exactly the version of the running kernel module, otherwise programs
#    fail with "driver/library version mismatch". Check before installing.
running=$(sed -n 's/.*Kernel Module *\([0-9.]*\).*/\1/p' /proc/driver/nvidia/version)
candidate=$(apt-cache policy "$PKG" | sed -n 's/ *Candidate: *\([0-9.]*\)-.*/\1/p')
echo "running NVIDIA kernel module: $running"
echo "package $PKG candidate:   $candidate"
if [ -z "$candidate" ] || [ "$running" != "$candidate" ]; then
    echo "Versions differ (or package not found). Stopping without changing anything."
    exit 1
fi

# 2. Install. Show what apt plans to do first (should be only libnvidia-encode + small deps).
sudo apt-get update
apt-get install --simulate "$PKG" | grep -E '^(Inst|Remv)' || true
read -r -p "Install the packages listed above? [y/N] " answer
if [ "$answer" != "y" ] && [ "$answer" != "Y" ]; then
    echo "Cancelled."
    exit 1
fi
sudo apt-get install -y "$PKG"

# 3. Docker reaches the GPUs through a list of driver files (the CDI spec). Regenerate it so
#    containers can use the encoder too (running containers are not affected).
sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml

# 4. Quick check: encode one second of a test picture on the GPU.
echo
echo "== NVENC test"
ffmpeg -hide_banner -v error -f lavfi -i testsrc=size=584x480:rate=60 -t 1 -c:v h264_nvenc -f null - \
    && echo "NVENC works" || echo "NVENC test FAILED"
echo
echo "Done. Tell Claude it's installed; it will compare NVENC with x264 on a sample game."
