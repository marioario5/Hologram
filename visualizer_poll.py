#!/usr/bin/env python3
"""
Hologram Visualizer — Python Polling Daemon (Full SPI Edition)
All input/output flows through SPI. UART removed entirely.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODE ARCHITECTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Two completely independent modes with their own state:

  SPOTIFY MODE  — driven by spotify_thread()
    • Polls Spotify API every 2s for currently playing track
    • On track change: downloads art, extracts palette, randomises plasma
    • Owns: spotify_state (last_track_id, album_art_bytes, palette, params)
    • Never touches vinyl_state

  VINYL MODE  — driven by spi_thread() + fingerprint_thread()
    • ESP32 signals needle drop via PKT_STATUS bit 0
    • RMS byte in PKT_STATUS[1] acts as noise gate — below threshold,
      no audio is collected and ffmpeg is never invoked
    • Audio chunks arrive via PKT_AUDIO → vinyl_audio_queue
    • fingerprint_thread: capture → RMS check → ffmpeg → DB match
    • Owns: vinyl_state (last_track_id, album_art_bytes, palette, params,
                         track_start_time, track_duration_ms)
    • Never touches spotify_state

  write_shm() picks the active state based on current mode. Neither
  mode can corrupt the other's state.

  LEARN MODE  — the only overlap, triggered by:
    • Flask POST /learn  (from local web UI)
    • PKT_LEARN from ESP32 hardware button
    Runs _learn_thread() which:
      1. Tees audio into learn_audio_queue (does NOT drain vinyl_audio_queue)
      2. Noise gate check — aborts if signal is silent
      3. ffmpeg + fingerprint in parallel with Spotify metadata fetch
      4. Writes ONLY to the fingerprint DB
      5. Does NOT modify vinyl_state or spotify_state

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NOISE GATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  The ESP32 sends a normalised RMS byte (0–255) in PKT_STATUS[1],
  computed on-device before any audio is transmitted.

  • Below NOISE_GATE_RMS: audio_live=False, no PKT_AUDIO polls,
    vinyl_audio_queue stays empty, ffmpeg is never spawned.
  • Above NOISE_GATE_RMS: audio_live=True, chunks collected,
    fingerprinting starts.

  A second software RMS check on the captured PCM buffer guards
  against the gate opening briefly on a transient and then closing —
  if the buffer itself is mostly silence, ffmpeg is still skipped.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SPI PACKET PROTOCOL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  All packets: [TYPE][SEQ][LEN_HI][LEN_LO][...PAYLOAD...][XOR]

  ESP32 → Pi (inbound):
    PKT_STATUS  0x01  6 bytes:
                  [0] bit0=mode(0=Spotify,1=Vinyl)  bit1=learn_pending
                  [1] RMS energy byte 0–255 (noise gate source)
                  [2–5] reserved
    PKT_AUDIO   0x02  512 bytes – raw s16le mono PCM @ 22050 Hz
    PKT_BANDS   0x03  17 bytes – 16× uint8 band levels + 1× uint8 BPM
    PKT_LEARN   0x04  0 bytes  – hardware learn button pressed

  Pi → ESP32 (outbound):
    CMD_POLL_STATUS  0x81  – request status
    CMD_POLL_AUDIO   0x82  – request audio chunk
    CMD_POLL_BANDS   0x83  – request band data
    CMD_ACK_LEARN    0x84  – acknowledge learn trigger
"""

import os, time, random, struct, mmap, threading
import subprocess, queue, sqlite3, logging, requests
import numpy as np
import spidev
from io import BytesIO
from colorthief import ColorThief
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from PIL import Image
from flask import Flask, jsonify, abort

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
log = logging.getLogger('hologram')

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()
CLIENT_ID     = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URL  = os.getenv("REDIRECT_URL", "http://127.0.0.1:8888/callback")

SPI_BUS      = 0
SPI_DEVICE   = 0
SPI_SPEED_HZ = 4_000_000

DISPLAY_W  = 1024
DISPLAY_H  = 600
NUM_BANDS  = 16
ALBUM_SIZE = 180
FLASK_PORT = 5000

SAMPLE_RATE   = 22050
CAPTURE_SECS  = 15
CAPTURE_BYTES = SAMPLE_RATE * 2 * CAPTURE_SECS   # s16le mono

DB_PATH = '/home/marioario/fingerprints.db'
ART_DIR = '/home/marioario/album_art/'

# Noise gate: ESP32 sends normalised RMS (0–255) in PKT_STATUS[1].
# Below this, no audio is collected and ffmpeg is never invoked.
NOISE_GATE_RMS = 18        # ~7% of ESP32 full scale — tune to taste

# Second gate: software RMS on raw captured PCM (s16le, range 0–32767).
# Guards against the ESP32 gate opening on a transient then closing,
# leaving a mostly-silent buffer.
NOISE_GATE_PCM_RMS = 200   # ~0.6% of s16le full scale

MATCH_THRESH = 0.35        # Hamming distance — lower is stricter

# ── SPI packet constants ──────────────────────────────────────────────────────
PKT_STATUS = 0x01
PKT_AUDIO  = 0x02
PKT_BANDS  = 0x03
PKT_LEARN  = 0x04

CMD_POLL_STATUS = 0x81
CMD_POLL_AUDIO  = 0x82
CMD_POLL_BANDS  = 0x83
CMD_ACK_LEARN   = 0x84

MODE_SPOTIFY = 0x00
MODE_VINYL   = 0x01

# ── Shared memory layout (must match C renderer) ──────────────────────────────
SHM_PATH  = "/tmp/hologram.shm"
SHM_SIZE  = 128 * 1024
MAGIC_VAL = 0xDEADBEEF

OFF_MAGIC        = 0
OFF_SEQ          = 4
OFF_MODE         = 8
OFF_SHOW_ALBUM   = 12
OFF_SHOW_BANDS   = 16
OFF_ALBUM_SIZE   = 20
# offset 24 reserved (was NUM_LEDS)
OFF_BPM          = 28
OFF_BANDS        = 32    # 16 × float32 = 64 bytes
OFF_PALETTE      = 96    # 5 × 3 uint8
OFF_PLASMA_ABCD  = 156   # 4 × float32
OFF_PLASMA_CXCY  = 172   # 2 × float32
OFF_STALA_COLORS = 180   # 7 × float32
OFF_ALBUM_READY  = 208
OFF_ALBUM_DATA   = 212

# ── Shared memory ─────────────────────────────────────────────────────────────
def open_shm():
    fd = os.open(SHM_PATH, os.O_CREAT | os.O_RDWR, 0o666)
    os.ftruncate(fd, SHM_SIZE)
    m = mmap.mmap(fd, SHM_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    os.close(fd)
    return m

shm = open_shm()

def shm_u32(off, v):   shm[off:off+4] = struct.pack('<I', int(v) & 0xFFFFFFFF)
def shm_f32s(off, a):  shm[off:off+len(a)*4] = struct.pack(f'<{len(a)}f', *a)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STATE — two completely separate dicts, one per mode
# Neither mode ever reads or writes the other's dict.
# write_shm() picks one based on active_mode.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEFAULT_PALETTE = [(255,0,128),(128,0,255),(0,128,255),(0,255,200),(255,255,100)]

def _blank_display_state():
    return dict(
        album_art_bytes = None,
        palette         = list(DEFAULT_PALETTE),
        last_track_id   = None,
        a  = random.uniform(.3, .7),
        b  = random.uniform(.1, .4),
        c  = random.uniform(.4, .8),
        d  = random.uniform(.05, .2),
        cx = random.uniform(np.pi * 0.4, np.pi * 1.6),  # keep center away from edges
        cy = random.uniform(np.pi * 0.4, np.pi * 1.6),
    )

# Spotify-owned display state — written only by spotify_thread()
spotify_state      = _blank_display_state()
spotify_state_lock = threading.Lock()

# Vinyl-owned display state — written only by _apply_vinyl_match()
vinyl_state      = _blank_display_state()
vinyl_state_lock = threading.Lock()

# Vinyl predictive timer — written only by spi_thread / _fingerprint_thread
vinyl_track_start_time  = 0.0
vinyl_track_duration_ms = 0

# Shared UI preferences and band data (written by spi_thread, read by shm_loop)
ui_state = dict(
    mode              = 'plasma',
    show_album        = True,
    show_bands        = True,
    album_size        = ALBUM_SIZE,
    bpm               = 120,
    bands             = [128] * NUM_BANDS,
    stalagmite_colors = [0.0] * 7,
    seq               = 0,
)
ui_state_lock = threading.Lock()

# Active hardware mode — set exclusively by spi_thread
active_mode      = MODE_SPOTIFY
active_mode_lock = threading.Lock()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AUDIO QUEUES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Filled by spi_thread when in vinyl mode AND above noise gate.
# Consumed by _fingerprint_thread for track identification.
vinyl_audio_queue = queue.Queue()

# Filled by spi_thread when is_learning is True (tee of the live stream).
# Consumed ONLY by _learn_thread. Never competes with vinyl_audio_queue.
learn_audio_queue = queue.Queue()

# ── Concurrency guards ────────────────────────────────────────────────────────
_fp_lock      = threading.Lock()   # prevents two fingerprint sessions
_learn_lock   = threading.Lock()   # prevents two learn sessions
is_fingerprinting = False
is_learning       = False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def new_plasma_params():
    return dict(
        a  = random.uniform(.3, .7),
        b  = random.uniform(.1, .4),
        c  = random.uniform(.4, .8),
        d  = random.uniform(.05, .2),
        cx = random.uniform(np.pi * 0.4, np.pi * 1.6),  # keep center away from edges
        cy = random.uniform(np.pi * 0.4, np.pi * 1.6),
    )

def extract_palette(img_bytes, n=5):
    img_bytes.seek(0)
    p = ColorThief(img_bytes).get_palette(color_count=n, quality=1)
    return sorted(p, key=lambda c: sum(c) / 3)

def download_art(url):
    return BytesIO(requests.get(url, timeout=5).content)

def drain_queue(q):
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break

def pcm_rms(raw_bytes: bytes) -> float:
    """RMS amplitude of s16le PCM, returned in range 0–32767."""
    if len(raw_bytes) < 2:
        return 0.0
    samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples ** 2)))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SPI PACKET BUILDER / PARSER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_command(cmd_type, payload: bytes = b'') -> list[int]:
    length = len(payload)
    header = [cmd_type, 0x00, (length >> 8) & 0xFF, length & 0xFF]
    body   = header + list(payload)
    xor    = 0
    for b in body:
        xor ^= b
    return body + [xor]

CMD_STATUS_PKT = build_command(CMD_POLL_STATUS)
CMD_AUDIO_PKT  = build_command(CMD_POLL_AUDIO)
CMD_BANDS_PKT  = build_command(CMD_POLL_BANDS)
CMD_ACK_PKT    = build_command(CMD_ACK_LEARN)

def pad_to(cmd: list[int], length: int) -> list[int]:
    return (cmd + [0x00] * length)[:length]

AUDIO_PAYLOAD   = 512
STATUS_RESP_LEN = 4 + 6 + 1       # header + 6-byte payload + checksum
AUDIO_RESP_LEN  = 4 + AUDIO_PAYLOAD + 1
BANDS_RESP_LEN  = 4 + 17 + 1      # 16 bands + 1 BPM
LEARN_RESP_LEN  = 4 + 0 + 1       # empty payload


class PacketParser:
    def __init__(self):
        self.expected = 0
        self.dropped  = 0

    def parse(self, raw: list[int]):
        if len(raw) < 5:
            return None
        ptype  = raw[0]
        seq    = raw[1]
        length = (raw[2] << 8) | raw[3]
        if len(raw) < 4 + length + 1:
            return None
        payload  = raw[4:4 + length]
        checksum = raw[4 + length]
        xor = 0
        for b in raw[:4 + length]:
            xor ^= b
        if xor != checksum:
            return None
        if seq != self.expected:
            self.dropped += (seq - self.expected) & 0xFF
        self.expected = (seq + 1) & 0xFF
        return (ptype, bytes(payload))


parser = PacketParser()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SPI THREAD — sole owner of the SPI bus
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def spi_thread():
    global active_mode, vinyl_track_start_time, vinyl_track_duration_ms

    try:
        spi = spidev.SpiDev()
        spi.open(SPI_BUS, SPI_DEVICE)
        spi.max_speed_hz = SPI_SPEED_HZ
        spi.mode = 0
    except Exception as e:
        log.error(f"SPI open failed: {e}")
        return

    last_poll  = 0.0
    prev_mode  = MODE_SPOTIFY
    audio_live = False   # True only when vinyl + above noise gate

    while True:
        now = time.time()
        if now - last_poll < 0.05:
            time.sleep(0.005)
            continue
        last_poll = now

        try:
            # ── 1. Status poll ────────────────────────────────────────────────
            raw    = spi.xfer2(pad_to(CMD_STATUS_PKT, STATUS_RESP_LEN), SPI_SPEED_HZ, 10)
            result = parser.parse(raw)

            if result and result[0] == PKT_STATUS:
                payload       = result[1]
                esp_mode      = payload[0] & 0x01         # bit 0: mode
                learn_pending = bool(payload[0] & 0x02)   # bit 1: learn queued
                rms_byte      = payload[1]                 # ESP32 normalised RMS

                # ── Mode transitions ──────────────────────────────────────────
                if esp_mode != prev_mode:
                    prev_mode = esp_mode
                    with active_mode_lock:
                        active_mode = esp_mode
                    if esp_mode == MODE_VINYL:
                        log.info("VINYL_START: Needle drop detected.")
                        drain_queue(vinyl_audio_queue)
                    else:
                        log.info("VINYL_STOP: Needle lifted / silence.")
                        audio_live = False
                        vinyl_track_start_time = 0.0

                # ── Noise gate (vinyl only) ───────────────────────────────────
                # The gate is computed by the ESP32 from what it's already
                # buffering. We never transfer audio unless it passes.
                if esp_mode == MODE_VINYL:
                    above_gate = (rms_byte >= NOISE_GATE_RMS)
                    if above_gate and not audio_live:
                        log.info(f"Noise gate OPEN (ESP RMS={rms_byte}). Starting capture.")
                        audio_live = True
                        _trigger_fingerprint()
                    elif not above_gate and audio_live:
                        log.info(f"Noise gate CLOSED (ESP RMS={rms_byte}). Pausing capture.")
                        audio_live = False

                # ── Learn button ──────────────────────────────────────────────
                if learn_pending:
                    spi.xfer2(pad_to(CMD_ACK_PKT, LEARN_RESP_LEN), SPI_SPEED_HZ, 10)
                    _start_learn("ESP32 hardware button")

            # ── 2. Predictive timer (vinyl only) ─────────────────────────────
            with active_mode_lock:
                current_mode = active_mode
            if (current_mode == MODE_VINYL
                    and vinyl_track_start_time > 0
                    and not is_fingerprinting):
                elapsed = now - vinyl_track_start_time
                if elapsed > (vinyl_track_duration_ms / 1000.0) + 8:
                    log.info("Predictive trigger: Expected end of track. Re-listening...")
                    vinyl_track_start_time = 0.0
                    drain_queue(vinyl_audio_queue)
                    _trigger_fingerprint()

            # ── 3. Band data poll (both modes — drives visualiser) ────────────
            raw_bands = spi.xfer2(pad_to(CMD_BANDS_PKT, BANDS_RESP_LEN), SPI_SPEED_HZ, 10)
            res_bands = parser.parse(raw_bands)
            if res_bands and res_bands[0] == PKT_BANDS and len(res_bands[1]) >= 17:
                bp    = res_bands[1]
                bands = [int(b) for b in bp[:NUM_BANDS]]
                bpm   = max(60, min(200, int(bp[16])))
                with ui_state_lock:
                    ui_state['bands'] = bands
                    ui_state['bpm']   = bpm

            # ── 4. Audio poll — ONLY when vinyl is active AND above noise gate ─
            # If audio_live is False this branch is entirely skipped.
            # No SPI transfer, no queue fill, no ffmpeg ever spawned.
            if audio_live:
                raw_audio = spi.xfer2(pad_to(CMD_AUDIO_PKT, AUDIO_RESP_LEN), SPI_SPEED_HZ, 10)
                res_audio = parser.parse(raw_audio)
                if res_audio and res_audio[0] == PKT_AUDIO:
                    chunk = bytes(res_audio[1])
                    vinyl_audio_queue.put(chunk)
                    # Tee into learn queue only while a learn session is running
                    if is_learning:
                        learn_audio_queue.put(chunk)

        except Exception as e:
            log.warning(f"SPI error: {e}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VINYL FINGERPRINT — identification only, zero Spotify involvement
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _trigger_fingerprint():
    if not _fp_lock.acquire(blocking=False):
        return   # already running
    _fp_lock.release()
    threading.Thread(target=_fingerprint_thread, daemon=True, name="FP").start()

def _fingerprint_thread():
    global is_fingerprinting, vinyl_track_start_time, vinyl_track_duration_ms

    if not _fp_lock.acquire(blocking=False):
        return
    is_fingerprinting = True
    try:
        log.info(f"Vinyl: capturing {CAPTURE_SECS}s for identification...")
        pcm      = bytearray()
        deadline = time.time() + CAPTURE_SECS + 5

        while len(pcm) < CAPTURE_BYTES:
            if time.time() > deadline:
                break
            try:
                pcm.extend(vinyl_audio_queue.get(timeout=1.0))
            except queue.Empty:
                continue

        if len(pcm) < CAPTURE_BYTES:
            log.warning("Vinyl: insufficient audio — aborting identification.")
            return

        # Software noise gate: if the buffer itself is mostly silence
        # (e.g. ESP gate opened briefly on a transient), skip ffmpeg entirely.
        rms = pcm_rms(bytes(pcm[:CAPTURE_BYTES]))
        if rms < NOISE_GATE_PCM_RMS:
            log.info(f"Vinyl: buffer is silent (PCM RMS={rms:.0f}) — skipping ffmpeg.")
            return

        raw_path = '/tmp/vinyl.raw'
        wav_path = '/tmp/vinyl.wav'
        with open(raw_path, 'wb') as f:
            f.write(bytes(pcm[:CAPTURE_BYTES]))

        log.info("Vinyl: running ffmpeg...")
        subprocess.run(
            ['ffmpeg', '-y', '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', '1',
             '-i', raw_path, '-threads', '1', wav_path],
            capture_output=True)

        try:
            import acoustid
            _, fp = acoustid.fingerprint_file(wav_path)
            match = _db_match(fp)
            if match:
                log.info(f"Vinyl MATCH: {match['artist']} – {match['title']} "
                         f"(dist={match['dist']:.3f})")
                _apply_vinyl_match(match)
                vinyl_track_start_time  = time.time()
                vinyl_track_duration_ms = match['duration_ms']
            else:
                log.info("Vinyl: no match found in local DB.")
        except Exception as e:
            log.error(f"Vinyl fingerprint error: {e}")

        for p in [raw_path, wav_path]:
            try:
                os.remove(p)
            except OSError:
                pass

    finally:
        is_fingerprinting = False
        _fp_lock.release()


def _apply_vinyl_match(match):
    """Update vinyl_state only. Never touches spotify_state."""
    img = None
    if match.get('art_path') and os.path.exists(match['art_path']):
        with open(match['art_path'], 'rb') as f:
            img = BytesIO(f.read())
    palette = extract_palette(img) if img else list(DEFAULT_PALETTE)
    with vinyl_state_lock:
        vinyl_state['album_art_bytes'] = img
        vinyl_state['palette']         = palette
        vinyl_state['last_track_id']   = match['id']
        vinyl_state.update(new_plasma_params())

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LEARN THREAD — the only place vinyl audio and Spotify metadata meet
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _start_learn(source: str):
    global is_learning
    if not _learn_lock.acquire(blocking=False):
        log.warning(f"Learn requested ({source}) but already in progress — ignoring.")
        return
    is_learning = True
    drain_queue(learn_audio_queue)
    log.info(f"LEARN MODE STARTED (triggered by: {source})")
    threading.Thread(target=_learn_thread, daemon=True, name="Learn").start()


def _learn_thread():
    """
    The ONLY function that touches both vinyl audio and Spotify at the same time.
    Writes exclusively to the fingerprint DB — does not modify vinyl_state or
    spotify_state.
    """
    global is_learning
    try:
        # ── Step 1: Capture audio ─────────────────────────────────────────────
        # Receives from learn_audio_queue, which is tee'd by spi_thread from the
        # live vinyl stream. vinyl_audio_queue is untouched, so identification
        # can continue running in parallel if needed.
        log.info(f"Learn: capturing {CAPTURE_SECS}s...")
        pcm      = bytearray()
        deadline = time.time() + CAPTURE_SECS + 8

        while len(pcm) < CAPTURE_BYTES:
            if time.time() > deadline:
                break
            try:
                pcm.extend(learn_audio_queue.get(timeout=1.0))
            except queue.Empty:
                continue

        if len(pcm) < CAPTURE_BYTES:
            log.error("Learn: not enough audio — is the needle down?")
            return

        # Noise gate on the captured buffer
        rms = pcm_rms(bytes(pcm[:CAPTURE_BYTES]))
        if rms < NOISE_GATE_PCM_RMS:
            log.error(f"Learn: captured audio is silent (PCM RMS={rms:.0f}) — aborting.")
            return

        # ── Step 2: Fetch Spotify metadata in parallel ────────────────────────
        # This is the ONLY moment Spotify and vinyl share execution context.
        # The result goes to the DB, not to any live state.
        spotify_result = {}
        spotify_error  = [None]

        def _fetch_spotify():
            try:
                sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                client_id=CLIENT_ID, client_secret=CLIENT_SECRET,
                redirect_uri=REDIRECT_URL, scope="user-read-currently-playing user-read-playback-state",
                open_browser=False,
                cache_path='/home/marioario/.spotify_cache'))
                pb = sp.current_playback()
                if not pb or not pb.get('item'):
                    spotify_error[0] = "Nothing playing on Spotify"
                    return
                item      = pb['item']
                tid       = item['id']
                art_url   = item['album']['images'][0]['url']
                local_art = os.path.join(ART_DIR, f"{tid}.jpg")
                with open(local_art, 'wb') as f:
                    f.write(requests.get(art_url, timeout=5).content)
                spotify_result.update(dict(
                    id          = tid,
                    artist      = item['artists'][0]['name'],
                    title       = item['name'],
                    duration_ms = item['duration_ms'],
                    art_path    = local_art,
                ))
            except Exception as e:
                spotify_error[0] = str(e)

        sp_thread = threading.Thread(target=_fetch_spotify, daemon=True)
        sp_thread.start()

        # ── Step 3: Convert + fingerprint (runs while Spotify fetch is in flight)
        raw_path = '/tmp/learn.raw'
        wav_path = '/tmp/learn.wav'
        with open(raw_path, 'wb') as f:
            f.write(bytes(pcm[:CAPTURE_BYTES]))

        log.info("Learn: running ffmpeg...")
        subprocess.run(
            ['ffmpeg', '-y', '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', '1',
             '-i', raw_path, '-threads', '1', wav_path],
            capture_output=True)

        import acoustid
        _, fp = acoustid.fingerprint_file(wav_path)

        for p in [raw_path, wav_path]:
            try:
                os.remove(p)
            except OSError:
                pass

        # ── Step 4: Wait for Spotify result and write DB ──────────────────────
        sp_thread.join(timeout=10)

        if spotify_error[0]:
            log.error(f"Learn: Spotify fetch failed — {spotify_error[0]}")
            return
        if not spotify_result:
            log.error("Learn: Spotify returned no metadata.")
            return

        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            'INSERT OR REPLACE INTO tracks '
            '(id, artist, title, album_art_path, fingerprint, duration_ms) '
            'VALUES (?,?,?,?,?,?)',
            (spotify_result['id'], spotify_result['artist'], spotify_result['title'],
             spotify_result['art_path'], fp, spotify_result['duration_ms']))
        conn.commit()
        conn.close()
        log.info(f"Learn: stored '{spotify_result['artist']} – {spotify_result['title']}' "
                 f"({spotify_result['duration_ms']/1000:.1f}s)")

    except Exception as e:
        log.error(f"Learn: unexpected error — {e}")
    finally:
        is_learning = False
        _learn_lock.release()
        log.info("LEARN MODE ENDED")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATABASE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _db_match(fingerprint):
    if not os.path.exists(DB_PATH):
        return None
    try:
        fp_arr    = _fp_to_array(fingerprint)
        conn      = sqlite3.connect(DB_PATH)
        rows      = conn.execute(
            'SELECT id, artist, title, album_art_path, fingerprint, duration_ms FROM tracks'
        ).fetchall()
        conn.close()

        best      = None
        best_dist = float('inf')
        for row in rows:
            dist = _hamming(fp_arr, _fp_to_array(row[4]))
            if dist < best_dist:
                best_dist = dist
                best = dict(id=row[0], artist=row[1], title=row[2],
                            art_path=row[3], duration_ms=row[5], dist=dist)

        return best if best and best_dist < MATCH_THRESH else None
    except Exception as e:
        log.error(f"DB match error: {e}")
        return None


def _fp_to_array(fp_str):
    import chromaprint
    ints, _ = chromaprint.decode_fingerprint(fp_str)
    return np.array(ints, dtype=np.uint32)


def _hamming(a, b):
    n   = min(len(a), len(b))
    xor = np.bitwise_xor(a[:n], b[:n])
    return int(np.sum(np.unpackbits(xor.view(np.uint8)))) / (n * 32)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SPOTIFY THREAD — completely isolated, never reads or writes vinyl_state
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def spotify_thread():
    """
    Polls Spotify every 2s. Writes only to spotify_state.
    While in vinyl mode it does nothing at all — it does not
    pause/resume logic around vinyl, it simply skips its update loop.
    """
    try:
        sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=CLIENT_ID, client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URL, scope="user-read-currently-playing user-read-playback-state",
        open_browser=False,
        cache_path='/home/marioario/.spotify_cache'))
    except Exception as e:
        log.error(f"Spotify auth failed: {e}")
        return

    poll_interval = 2
    while True:
        with active_mode_lock:
            mode = active_mode
        if mode == MODE_VINYL:
            time.sleep(2)
            continue
        try:
            pb = sp.current_playback()
            if pb and pb.get('is_playing') and pb.get('item'):
                item = pb['item']
                tid  = item['id']
                with spotify_state_lock:
                    last = spotify_state['last_track_id']
                if tid != last:
                    log.info(f"Spotify: {item['artists'][0]['name']} – {item['name']}")
                    img = download_art(item['album']['images'][0]['url'])
                    with spotify_state_lock:
                        spotify_state['last_track_id']   = tid
                        spotify_state['album_art_bytes'] = img
                        spotify_state['palette']         = extract_palette(img)
                        spotify_state.update(new_plasma_params())
                    poll_interval = 1  # poll faster right after a change
                else:
                    poll_interval = 2  # back to normal
        except Exception as e:
            log.warning(f"Spotify poll error: {e}")
        time.sleep(poll_interval)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STALAGMITE THREAD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def stalagmite_thread():
    pass  # unchanged

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SHM WRITER — picks the correct state based on active mode, never mixes them
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_last_art_id = None
def _write_shm():
    global _last_art_id
    with active_mode_lock:
        mode = active_mode

    if mode == MODE_VINYL:
        with vinyl_state_lock:
            ds = dict(vinyl_state)
    else:
        with spotify_state_lock:
            ds = dict(spotify_state)

    with ui_state_lock:
        ui = dict(ui_state)
        ui['seq'] = (ui['seq'] + 1) & 0xFFFFFFFF
        ui_state['seq'] = ui['seq']

    shm_f32s(OFF_BANDS, [float(b) for b in ui['bands']])
    for i, c in enumerate(ds['palette']):
        shm[OFF_PALETTE + i*3 : OFF_PALETTE + i*3 + 3] = bytes(c)
    shm_f32s(OFF_PLASMA_ABCD,  [ds['a'], ds['b'], ds['c'], ds['d']])
    shm_f32s(OFF_PLASMA_CXCY,  [ds['cx'], ds['cy']])
    shm_f32s(OFF_STALA_COLORS, ui['stalagmite_colors'])
    shm_u32(OFF_MODE,       0 if ui['mode'] == 'plasma' else 1)
    shm_u32(OFF_SHOW_ALBUM, 1 if ui['show_album'] else 0)
    shm_u32(OFF_SHOW_BANDS, 1 if ui['show_bands'] else 0)
    shm_u32(OFF_ALBUM_SIZE, ui['album_size'])
    shm_u32(OFF_BPM,        ui['bpm'])

    if ds['album_art_bytes']:
        current_id = ds.get('last_track_id')
        if current_id != _last_art_id:
            shm_u32(OFF_ALBUM_READY, 0)
            ds['album_art_bytes'].seek(0)
            sz = ui['album_size']
            img = (Image.open(ds['album_art_bytes'])
                   .convert('RGB')
                   .resize((sz, sz), Image.LANCZOS))
            shm[OFF_ALBUM_DATA : OFF_ALBUM_DATA + sz * sz * 3] = img.tobytes()
            _last_art_id = current_id
            shm_u32(OFF_ALBUM_READY, 1)
    else:
        shm_u32(OFF_ALBUM_READY, 0)

    shm_u32(OFF_SEQ,   ui['seq'])
    shm_u32(OFF_MAGIC, MAGIC_VAL)


def shm_loop():
    while True:
        _write_shm()
        time.sleep(1 / 60)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FLASK — /status (read-only) + /learn (triggers learning mode)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

app = Flask(__name__)


@app.route('/status')
def route_status():
    with active_mode_lock:
        mode = active_mode
    with ui_state_lock:
        bpm = ui_state['bpm']
    if mode == MODE_VINYL:
        with vinyl_state_lock:
            track = vinyl_state['last_track_id']
    else:
        with spotify_state_lock:
            track = spotify_state['last_track_id']
    return jsonify(dict(
        mode           = 'vinyl' if mode == MODE_VINYL else 'spotify',
        track          = track,
        bpm            = bpm,
        learning       = is_learning,
        fingerprinting = is_fingerprinting,
    ))


@app.route('/learn', methods=['POST'])
def route_learn():
    """Trigger learning mode from the local web UI."""
    with active_mode_lock:
        mode = active_mode
    if mode != MODE_VINYL:
        abort(409, description="Must be in vinyl mode to learn a track.")
    _start_learn("Flask /learn")
    return jsonify({'status': 'Learning started — capturing audio now...'})


def flask_thread():
    app.run(host='0.0.0.0', port=FLASK_PORT, debug=False, use_reloader=False)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENTRYPOINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == '__main__':
    log.info("=== Hologram Visualizer Daemon ===")
    shm_u32(OFF_MAGIC, 0)
    os.makedirs(ART_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        'CREATE TABLE IF NOT EXISTS tracks '
        '(id TEXT PRIMARY KEY, artist TEXT, title TEXT, '
        ' album_art_path TEXT, fingerprint TEXT, duration_ms INTEGER)')
    conn.commit()
    conn.close()

    for name, target in [
        ("SPI",     spi_thread),       # bus owner — status, bands, audio, learn ACK
        ("Spotify", spotify_thread),   # Spotify polling — writes only spotify_state
        ("Stala",   stalagmite_thread),
        ("Flask",   flask_thread),     # /status + POST /learn
        ("SHM",     shm_loop),         # 60 Hz SHM writer
    ]:
        threading.Thread(target=target, daemon=True, name=name).start()
        log.info(f"Started: {name}")

    while True:
        time.sleep(1)