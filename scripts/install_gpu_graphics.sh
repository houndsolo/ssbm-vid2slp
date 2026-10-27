#!/usr/bin/env bash
# install_gpu_graphics.sh - add the OpenGL/Vulkan part of the NVIDIA driver, so Dolphin can
# draw on a GPU (GT 1030 or V100) instead of in software on the CPU.
#
# Why: the server has the compute-only ("headless") NVIDIA driver 580.178.04. CUDA works,
# but graphics programs can't use the GPUs. The package libnvidia-gl-580-server adds just the
# graphics libraries for the SAME driver version. It does not replace the driver or the kernel
# module, so no reboot should be needed and CUDA jobs keep working.
#
# Run it yourself (needs sudo):
#     bash scripts/install_gpu_graphics.sh
set -euo pipefail

PKG=libnvidia-gl-580-server

# 1. The graphics libraries must be exactly the version of the running kernel module,
#    otherwise programs fail with "driver/library version mismatch". Check before installing.
running=$(sed -n 's/.*Kernel Module *\([0-9.]*\).*/\1/p' /proc/driver/nvidia/version)
candidate=$(apt-cache policy "$PKG" | sed -n 's/ *Candidate: *\([0-9.]*\)-.*/\1/p')
echo "running NVIDIA kernel module: $running"
echo "package $PKG candidate:      $candidate"
if [ -z "$candidate" ] || [ "$running" != "$candidate" ]; then
    echo "Versions differ (or package not found). Stopping without changing anything."
    exit 1
fi

# 2. Install. Show what apt plans to do first (should be only libnvidia-gl + small deps).
sudo apt-get update
apt-get install --simulate "$PKG" | grep -E '^(Inst|Remv)' || true
read -r -p "Install the packages listed above? [y/N] " answer
if [ "$answer" != "y" ] && [ "$answer" != "Y" ]; then
    echo "Cancelled."
    exit 1
fi
sudo apt-get install -y "$PKG"

# 3. Docker reaches the GPUs through a list of driver files (the CDI spec). Regenerate it so
#    it includes the new graphics libraries.
sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml

# 4. Quick checks.
echo
echo "== Vulkan driver file present?"
ls -l /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json 2>/dev/null || echo "not found (unexpected)"
echo "== CUDA still fine?"
nvidia-smi --query-gpu=index,name,driver_version --format=csv
echo
echo "Done. Tell Claude it's installed; it will test Dolphin rendering on the GT 1030."
