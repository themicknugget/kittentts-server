# Builds ONNX Runtime from source for optimized CPU inference.
# Source builds produce faster code than generic PyPI wheels (~1.7x).
# First build ~30–60 min; subsequent builds fast via Docker layer cache.

# Stage 1: compile ONNX Runtime
FROM ubuntu:22.04 AS ort-builder

ARG ORT_REF=v1.24.2
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git wget \
        python3 python3.10-dev python3-pip \
        libprotobuf-dev protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir "cmake<4" ninja numpy packaging

COPY eigen-src /tmp/eigen-src

WORKDIR /build

RUN git clone --depth 1 --branch ${ORT_REF} \
    https://github.com/microsoft/onnxruntime.git

RUN cd onnxruntime && ./build.sh \
        --allow_running_as_root \
        --config Release \
        --build_shared_lib \
        --parallel \
        --enable_pybind \
        --build_wheel \
        --skip_tests \
        --skip_submodule_sync \
        --cmake_extra_defines \
            onnxruntime_BUILD_UNIT_TESTS=OFF \
            FETCHCONTENT_SOURCE_DIR_EIGEN=/tmp/eigen-src

# Stage 2: lean runtime image
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=ort-builder /build/onnxruntime/build/Linux/Release/dist/*.whl /tmp/
COPY fix_execstack.py /tmp/fix_execstack.py
RUN pip install --no-cache-dir /tmp/onnxruntime*.whl && rm /tmp/onnxruntime*.whl \
    && python3 /tmp/fix_execstack.py '/usr/local/lib/python3.10/**/onnxruntime/**/*.so' \
    && rm /tmp/fix_execstack.py

RUN pip install --no-cache-dir --no-deps \
    "kittentts @ https://github.com/KittenML/KittenTTS/releases/download/0.8/kittentts-0.8.0-py3-none-any.whl" \
    && pip install --no-cache-dir \
    espeakng_loader huggingface_hub misaki phonemizer-fork num2words numpy soundfile spacy \
    fastapi \
    "uvicorn[standard]"

COPY server.py .

ENV KITTENTTS_MODEL=KittenML/kitten-tts-mini-0.8
ENV PORT=8080

EXPOSE 8080

CMD ["python", "server.py"]
