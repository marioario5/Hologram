#!/bin/bash
# setup.sh — Finalized for RPi 3B (32-bit Legacy + SPI Audio/Visuals)

set -e

echo "=== Hologram Visualizer Master Setup ==="

# 1. Enable Hardware SPI on the Raspberry Pi
echo "Enabling Hardware SPI..."
sudo raspi-config nonint do_spi 0

# 2. Install System Dependencies
echo "Updating system and installing hardware tools..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    libchromaprint-dev \
    libchromaprint-tools \
    ffmpeg \
    fbset \
    python3-pip \
    python3-dev \
    libjpeg-dev \
    zlib1g-dev

# 3. Configure Framebuffer (32-bit Legacy Mode) FLAG "sudo fbset -g 1024 600 1024 1200 32 || echo "fbset failed — ensure you are in legacy/fake-kms mode.""
echo "Configuring 1024x600 32-bit legacy framebuffer..."
sudo fbset -g 1920 1200 1920 2400 32 || echo "fbset failed — ensure you are in legacy/fake-kms mode."

# 4. Build Optimized C Renderer (NEON SIMD)
echo "Compiling NEON-optimized C renderer..."
gcc -O3 \
    -march=armv8-a+simd \
    -mtune=cortex-a53 \
    -ffast-math \
    -std=c11 \
    -o visualizer_render \
    visualizer_render.c \
    -lpthread \
    -lm

echo "Build complete: ./visualizer_render"

# 5. Install Python Dependencies
echo "Installing Python intelligence suite..."
pip3 install \
    spotipy \
    pillow \
    colorthief \
    flask \
    pyserial \
    numpy \
    python-dotenv \
    pyacoustid \
    requests \
    spidev \
    rpi_ws281x \
    --break-system-packages

echo ""
echo "=== Setup Successfully Finished ==="
echo "Note: Since SPI was enabled, you may need to REBOOT the Pi once before running!"
echo "1. Run Python Brain:  python3 visualizer_poll.py"
echo "2. Run C Muscle:      ./visualizer_render"