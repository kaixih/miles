# GB300 retains the successful upstream CUDA 13.0 image and native packages.
ARG BASE_IMAGE=radixark/miles@sha256:226f63d28e4b1482e0a6948ba3d486c1b1635648d079c82c9501640b24657986
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
COPY pr33743/pr33743-89f25a0.patch pr33743/manifest.json apply_trtllm_refit.py /opt/miles-moe/
RUN --network=none python3 -B /opt/miles-moe/apply_trtllm_refit.py --base-image "${BASE_IMAGE}"
COPY followup_idempotent_postload/ /opt/miles-moe/followup-idempotent-postload/
RUN --network=none python3 -B /opt/miles-moe/followup-idempotent-postload/apply_followup.py
LABEL ai.miles.extension="GB300 upstream BF16 TRTLLM hot-refit Python fixes" \
      ai.miles.sglang-pr="https://github.com/sgl-project/sglang/pull/33743" \
      ai.miles.sglang-pr-head="89f25a0b9772576fbd427a495a1144f544c90928"
