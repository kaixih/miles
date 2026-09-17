# Worker discovery transport recovery

A full Rubin run ended after generation completed because its cleanup queried
`GET /list_workers` and received `httpx.RemoteProtocolError`. The same router
process and all four engines remained alive; a subsequent fresh query returned
HTTP 200. This establishes the unhandled transport failure, not its low-level
cause. No recoverable training checkpoint remained after Ray teardown.

The scoped change makes only worker discovery opt into two transport retries
(three total attempts), with a ten-second per-request timeout and short bounded
backoff. HTTP error responses, invalid JSON, and task cancellation still
propagate. Other GET callers keep their existing no-retry behavior. Generation
POSTs, model weights, learning settings, numerical kernels, and graphs are
unchanged. Each retried failure is logged; persistent failures still stop the run.

`worker-discovery-transport.patch` records the exact difference from the original
frozen Miles runtime. The rerun must use a new immutable source snapshot and a
separate attempt ID; do not patch the original frozen snapshot in place or append
new results to the failed trajectory.

The CPU regression exercises the candidate GET function with the real `httpx`
client and a local TCP server that closes connections without a response:

```sh
python3 -B lab/rubin_trtllm_full/control_retry/test_control_retry.py \
  --source miles/utils/http_utils.py -v
```

The six cases cover successful recovery, exhaustion after three attempts,
unchanged default behavior, HTTP errors, invalid JSON, and invalid retry budgets.
