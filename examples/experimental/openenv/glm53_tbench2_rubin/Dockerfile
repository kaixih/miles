# Supply the private registry reference locally; do not put it in this file.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
USER root
WORKDIR /opt/miles

COPY provider.patch initial-snapshot-retry.patch /tmp/tb2-patches/
RUN test "$(git rev-parse HEAD)" = 0d9d726e0f9b9d487c36a0c342012b917820fb04 \
    && git apply --check /tmp/tb2-patches/provider.patch \
    && git apply /tmp/tb2-patches/provider.patch \
    && git apply --check /tmp/tb2-patches/initial-snapshot-retry.patch \
    && git apply /tmp/tb2-patches/initial-snapshot-retry.patch

# Relocate the existing checkout for non-root runtime users; do not upgrade it.
RUN cp -a /root/Megatron-LM /opt/Megatron-LM \
    && chmod -R a+rX /opt/Megatron-LM

COPY requirements-rubin.txt /tmp/tb2-requirements.txt
# Preserve the image's CUDA/PyTorch/TE/SGLang stack. This file is its observed
# package delta, not a requirements file for arbitrary base images.
RUN python3 -m pip install --no-deps -r /tmp/tb2-requirements.txt \
    && git clone https://github.com/huggingface/OpenEnv.git /opt/OpenEnv \
    && git -C /opt/OpenEnv checkout 38b2a31354d1d4d627894412b9d469e92f4a2c61 \
    && python3 -m pip install --no-deps --no-build-isolation \
        -e /opt/OpenEnv -e /opt/OpenEnv/envs/tbench2_env

RUN git clone https://github.com/laude-institute/terminal-bench-2.git /opt/terminal-bench-2 \
    && git -C /opt/terminal-bench-2 checkout 2fd12b88aafdd04a52c298e3940bcb189f9766d6 \
    && git config --system --add safe.directory /opt/terminal-bench-2 \
    && git config --system --add safe.directory /opt/OpenEnv \
    && git config --system --add safe.directory /opt/Megatron-LM \
    && git config --system --add safe.directory /opt/miles

COPY run_glm5_3_flash_tb2.py /opt/miles/examples/experimental/openenv/glm53_tbench2_rubin/
COPY train_tasks.txt eval_tasks.txt /opt/tb2-data/
ENV PYTHONPATH=/opt/Megatron-LM:/opt/miles/examples/experimental/openenv:/opt/miles
RUN python3 examples/experimental/openenv/make_tbench2_data.py \
        --tasks_dir /opt/terminal-bench-2 \
        --tasks "$(paste -sd, /opt/tb2-data/train_tasks.txt)" --output /opt/tb2-data/train69.jsonl \
    && python3 examples/experimental/openenv/make_tbench2_data.py \
        --tasks_dir /opt/terminal-bench-2 \
        --tasks "$(paste -sd, /opt/tb2-data/eval_tasks.txt)" --output /opt/tb2-data/eval20.jsonl \
    && cd /opt/tb2-data && sha256sum train69.jsonl eval20.jsonl > SHA256SUMS

# These are site bring-up helpers, not additional Miles source patches.
COPY rubin_node.py /opt/tb2-tools/rubin_node.py
