#!/usr/bin/env bash
# native_build.sh - build vid2slp's version of the native Melee engine host (headless, Linux).
#
# What: Melee Unlocked (github.com/houndsolo/melee-unlocked, commit fba2f21) + our patch
#   native/patches/0001-headless-replay-playback.patch  (Slippi replay playback with input/state
#   overrides for the headless host; see src/vid2slp/engine.py)
# Only the small headless HOST is rebuilt. The game library (melee_game.so, the game itself) is the
# existing build in data/tools/native-engine/build/game (same commit; it already contains Slippi's
# playback codes natively).
#
# Where: source in data/native/melee-unlocked (gitignored), build in data/native/build/headless.
# Needs: Docker image vid2slp-dev:gcc14 (g++-14) and cmake/ninja (pip packages, installed into
#   data/native/buildtools by this script if missing).
#
# Run:
#   bash scripts/native_build.sh            # build (clones and patches the source if needed)
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=data/native/melee-unlocked
COMMIT=fba2f21fbd37f30bed8dc97bfae68a074c8d4edc

if [ ! -d "$SRC" ]; then
  git clone --filter=blob:limit=2m https://github.com/houndsolo/melee-unlocked.git "$SRC"
  git -C "$SRC" checkout --detach "$COMMIT"
  git -C "$SRC" apply "$PWD/native/patches/0001-headless-replay-playback.patch"
fi
if [ ! -x data/native/buildtools/bin/ninja ]; then
  uv pip install --target data/native/buildtools "cmake>=3.25,<4" ninja
fi
mkdir -p data/native/build/headless
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD/data/native:/n" vid2slp-dev:gcc14 bash -c "
  /n/buildtools/cmake/data/bin/cmake -S /n/melee-unlocked/port/linux/headless -B /n/build/headless -G Ninja \
    -DCMAKE_MAKE_PROGRAM=/n/buildtools/bin/ninja -DCMAKE_CXX_COMPILER=g++-14 \
    -DCMAKE_BUILD_TYPE=Release -DMELEE_CPU_BASELINE=AVX2 > /n/build/configure.log &&
  /n/buildtools/cmake/data/bin/cmake --build /n/build/headless --target melee_headless"
ls -la data/native/build/headless/melee_headless
