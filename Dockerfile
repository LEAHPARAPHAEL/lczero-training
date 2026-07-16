FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

# Install system X11 GUI dependencies as root
RUN apt-get update && apt-get install -y \
    libx11-6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pygame globally so your non-root user can access it
RUN pip install --no-cache-dir pygame