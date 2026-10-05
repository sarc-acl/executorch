FROM docker.io/library/ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc-aarch64-linux-gnu g++-aarch64-linux-gnu binutils-aarch64-linux-gnu \
    cmake ninja-build git ca-certificates curl xz-utils python3 python3-venv \
    libvulkan-dev file pkg-config && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:/opt/vulkan/1.4.341.1/x86_64/bin:$PATH
RUN pip install --no-cache-dir pyyaml ruamel.yaml jinja2 numpy typing_extensions flatbuffers packaging \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN mkdir -p /opt/vulkan && curl -fL --retry 3 \
    https://sdk.lunarg.com/sdk/download/1.4.341.1/linux/vulkansdk-linux-x86_64-1.4.341.1.tar.xz \
    | tar -xJ -C /opt/vulkan
ENV LD_LIBRARY_PATH=/opt/vulkan/1.4.341.1/x86_64/lib
WORKDIR /work
