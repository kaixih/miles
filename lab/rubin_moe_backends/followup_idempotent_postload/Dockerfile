ARG PR33743_IMAGE=miles-rubin-trtllm-refit:parent-971195a84be7
FROM ${PR33743_IMAGE}
COPY manifest.json idempotent-postload.patch apply_followup.py test_real_flashinfer_postload.py /opt/miles-moe/followup-idempotent-postload/
RUN --network=none python3 -B /opt/miles-moe/followup-idempotent-postload/apply_followup.py
LABEL ai.miles.followup="Idempotent final BF16 TRTLLM post-load after PR33743 bucket repack"
