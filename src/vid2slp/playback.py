"""
playback.py - run Slippi Playback Dolphin headless and get back a "regenerated" replay.

Why: Playback Dolphin replays a .slp by feeding the recorded controller inputs into the real
game, which then simulates everything itself. With SlippiRegenerateReplays on, Dolphin writes
a NEW .slp of what actually happened. So if we put our own (predicted) inputs into a replay,
the regenerated file tells us exactly what the game does with them. Verified 2026-10-26: an
unmodified replay regenerates with 100% identical states.

How it runs: inside the Docker image `vid2slp-render:gcc14` (built by the old project) (it has the OpenGL
libraries this server lacks, plus Xvfb as a virtual screen). Needs:
  - --shm-size=2g  (with Docker's 64 MB default, Dolphin aborts immediately)
  - the Playback Dolphin 3.5.2 AppImage folder and the NTSC 1.02 ISO, mounted read-only
Dolphin does not exit by itself at the end frame, so we watch its "[CURRENT_FRAME] n" output
and stop it once the end frame has been reached. If the game ENDS earlier than the original
(e.g. predicted inputs lose a stock sooner), playback simply stops advancing; when no new frame
appears for `stall_seconds`, we stop and return what was regenerated so far.
A `timeout` inside the container is a backstop so a stuck container always ends on its own.
Note: Dolphin's log ends with a NUL byte, so use `grep -a` to search it by hand.
"""

import json
import os
import shutil
import signal
import subprocess
import time
import uuid

DOLPHIN_DIR = "/home/prompter/ssbm-vid2slp/data/tools/playback/squashfs-root"
ISO = "/home/prompter/ssbm-vid2slp/data/recordings/vanilla.iso"
IMAGE = "vid2slp-render:gcc14"

GECKO = "[Gecko_Enabled]\n$Required: General Codes\n$Required: Slippi Playback\n$Recommended: Slippi Recording\n"
DOLPHIN_INI = ("[Core]\nCPUThread = {dual}\nDSPHLE = True\nGFXBackend = {backend}\n"
               "SlippiRegenerateReplays = True\nSlippiReplayRegenerateDir = /out/regenerated\n"
               "{speed}[Interface]\nConfirmStop = False\n[DSP]\nBackend = No audio output\n{movie}")
GFX_INI = "[Hardware]\nVSync = False\n[Settings]\nShowFPS = False\n{dump}"
# Frame dumping (video rendering): Dolphin encodes every displayed frame with its built-in
# FFmpeg into user/Dump/Frames/. Used by scripts/m3_render.py.
MOVIE_DUMP = "[Movie]\nDumpFrames = True\nDumpFramesSilent = True\n"
GFX_DUMP = "DumpCodec = {codec}\nDumpFormat = {fmt}\nBitrateKbps = {kbps}\nInternalResolutionFrameDumps = False\n"


def regenerate(slp_path, out_dir, start=-123, end=None, timeout=900, unlimited_speed=True, resync=False,
               stall_seconds=60, dump_video=None, dual_core=False, gpu=None):
    """Play `slp_path` from frame `start` to `end` and return (regenerated .slp path, seconds taken).

    out_dir must not exist yet; it receives the Dolphin user folder, logs and the result.
    end=None plays to the end of the game (the last frame number is read from the file).
    resync=True turns on Slippi's desync correction during playback (normally off).
    dump_video: None (no video), or a dict like {"codec": "ffv1", "fmt": "avi", "kbps": 25000}
        to also write the rendered frames to out_dir/user/Dump/Frames/.
    dual_core: run the emulated CPU and GPU on separate threads (faster; only used for video).
    gpu: None = draw with software OpenGL on the CPU. Or an NVIDIA GPU number as nvidia-smi
        shows it (0 = GT 1030, 1 = V100) to draw with Vulkan on that GPU. Needs the host
        package libnvidia-gl-580-server (scripts/install_gpu_graphics.sh).
    """
    slp_path = os.path.abspath(slp_path)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(os.path.join(out_dir, "user", "Config"))
    os.makedirs(os.path.join(out_dir, "user", "GameSettings"))
    os.makedirs(os.path.join(out_dir, "regenerated"))
    os.makedirs(os.path.join(out_dir, "home"))
    open(os.path.join(out_dir, "user", "GameSettings", "GALE01r2.ini"), "w").write(GECKO)
    speed = "EmulationSpeed = 0.0\n" if unlimited_speed else ""
    movie = MOVIE_DUMP if dump_video else ""
    dump = GFX_DUMP.format(**dump_video) if dump_video else ""
    open(os.path.join(out_dir, "user", "Config", "Dolphin.ini"), "w").write(DOLPHIN_INI.format(speed=speed, movie=movie, dual=dual_core,
                                                                 backend="OGL" if gpu is None else "Vulkan"))
    open(os.path.join(out_dir, "user", "Config", "GFX.ini"), "w").write(GFX_INI.format(dump=dump))
    if end is None:
        import peppi_py
        end = peppi_py.read_slippi(slp_path, skip_frames=True).metadata["lastFrame"]
    request = {"mode": "normal", "replay": "/replay/game.slp", "commandId": f"vid2slp-{uuid.uuid4().hex[:8]}",
               "startFrame": start, "endFrame": end, "shouldResync": resync, "rollbackDisplayMethod": "off"}
    json.dump(request, open(os.path.join(out_dir, "request.json"), "w"))

    name = f"vid2slp-playback-{uuid.uuid4().hex[:10]}"
    # the GPU is handed to the container by NVIDIA's container toolkit (CDI device name)
    gpu_args = [] if gpu is None else ["--device", f"nvidia.com/gpu={gpu}", "-e", "NVIDIA_DRIVER_CAPABILITIES=all"]
    cmd = ["docker", "run", "--rm", "--init", "--name", name, "--shm-size=2g", "--network=none", *gpu_args,
           "-u", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/out/home",
           "-v", f"{DOLPHIN_DIR}:/playback:ro", "-v", f"{ISO}:/iso/melee.iso:ro",
           "-v", f"{slp_path}:/replay/game.slp:ro", "-v", f"{out_dir}:/out",
           IMAGE, "timeout", str(int(timeout)), "xvfb-run", "-a", "-s", "-screen 0 1024x768x24",
           "/playback/AppRun", "--exec=/iso/melee.iso", "--slippi-input=/out/request.json",
           "--user=/out/user", "--batch", "--cout"]
    log_path = os.path.join(out_dir, "stdout.log")
    t0 = time.time()
    reached = stalled = False
    last_size, last_progress = 0, time.time()
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None and time.time() - t0 < timeout + 30:
            time.sleep(0.5)
            size = os.path.getsize(log_path)
            with open(log_path, "rb") as f:
                f.seek(max(0, size - 200))
                tail = f.read().decode(errors="replace")
            if f"[CURRENT_FRAME] {end + 1}" in tail or f"[CURRENT_FRAME] {end}\n" in tail:
                reached = True
                time.sleep(2)  # let the recorder flush the last frames to disk
                break
            if size != last_size:
                last_size, last_progress = size, time.time()
            elif "[CURRENT_FRAME]" in tail and time.time() - last_progress > stall_seconds:
                stalled = True  # game ended early (or Dolphin hung): keep what we have
                break
        if proc.poll() is None:
            # `docker run` forwards SIGTERM to the container's init, which stops Dolphin.
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
    seconds = time.time() - t0
    files = sorted(os.listdir(os.path.join(out_dir, "regenerated")))
    if not (reached or stalled) or len(files) != 1:
        raise RuntimeError(f"playback failed (reached end: {reached}, files: {files}); see {log_path}")
    return os.path.join(out_dir, "regenerated", files[0]), seconds


def clean(out_dir):
    """Remove a playback work folder but keep nothing else (logs are small; call only if unwanted)."""
    shutil.rmtree(out_dir, ignore_errors=True)
