# Share perceptual encoder-cache metadata across DP schedulers

`ECMemcacheConnector` can use one shared pHash/SSIM index for all data-parallel
schedulers in a deployment. Memcache stores the resized tensors and encoder
embeddings; the index service stores pHashes, candidate keys, and ranked SSIM
scores. Hash computation, SSIM computation, and Memcache transfers remain in each
scheduler. No tensors are sent through the index RPC.

## Start vLLM normally

No separate service command is required. When a shared endpoint is configured,
the default `shared_index_mode: "embedded"` starts one background metadata thread
inside the serving DP0 scheduler's connector. Every scheduler, including DP0,
uses the same RPC interface. The thread creates and closes its own ZMQ socket;
it performs no tensor transfers, pHash/SSIM computation, or NPU initialization.
The normal vLLM environment supplies `pyzmq` and `msgpack`.

Ownership uses `parallel_config.data_parallel_index`, which remains the serving
rank even when dense models reset their compute DP rank to zero. The server is
not started in every process whose compute rank is zero.

DP0 binds the configured endpoint, so it must identify an interface on the DP0
host. For single-host DP, use `tcp://127.0.0.1:5558`. For multiple hosts, use a
reachable DP0 address such as `tcp://10.0.0.10:5558` in every scheduler. Use a free
port distinct from vLLM's DP coordination and API ports; do not use a wildcard
connect address. On Linux, `ipc:///absolute/path` is also supported for processes
on the same host. The endpoint is intended for a trusted deployment network;
the protocol does not provide authentication or encryption.

DP0 waits for the thread to bind before constructing its client. Other DPs may
start first: connection failures fall back to misses until DP0 becomes ready,
then the clients reconnect. A bind conflict fails DP0 initialization explicitly.

Closing DP0's connector stops and joins its metadata thread. Closing another
scheduler only closes its own client. If DP0 exits early or crashes, remaining
schedulers fall back to misses; there is no ownership election or independent
service process left running. A restarted DP0 starts with an empty index.

## Configure every DP identically

Add these settings to `ec_connector_extra_config.similarity_cache`:

```json
{
  "enabled": true,
  "matcher": "phash_ssim",
  "shared_index_endpoint": "tcp://127.0.0.1:5558",
  "shared_index_namespace": "qwen-vl-deployment",
  "shared_index_timeout_ms": 1000
}
```

Example connector configuration:

```bash
--ec-transfer-config '{
  "ec_connector": "ECMemcacheConnector",
  "ec_connector_module_path": "vllm_ascend.distributed.ec_transfer.ec_memcache_connector",
  "ec_role": "ec_both",
  "ec_connector_extra_config": {
    "similarity_cache": {
      "enabled": true,
      "shared_index_endpoint": "tcp://127.0.0.1:5558",
      "shared_index_namespace": "qwen-vl-deployment",
      "shared_index_timeout_ms": 1000
    }
  }
}'
```

| Setting | Default | Meaning |
| --- | --- | --- |
| `shared_index_endpoint` | `null` | `tcp://` or `ipc://` address; `null` preserves the process-local index. |
| `shared_index_namespace` | `"default"` | Nonempty deployment name, identical on every DP. |
| `shared_index_timeout_ms` | `1000` | Positive 32-bit integer; maximum wait for each send and receive, and reconnect cooldown. |
| `shared_index_mode` | `"embedded"` | DP0 owns a background server thread. Use `"external"` only for an independently managed server. |

Do not add the DP rank to the endpoint or namespace. All DPs must also access the
same Memcache data. Multiple DP schedulers without an endpoint still use separate
local indexes and emit a warning.

## Skip similarity matching for small images

Set `ec_connector_extra_config.similarity_cache.min_image_patches` to a
nonnegative integer. The default `0` disables this gate and preserves existing
behavior. For example, add `"min_image_patches": 256` to the `similarity_cache`
object to bypass similarity processing for images with fewer than 256 patches.
This is an illustrative threshold, not a hardware-tuned recommendation.

The count is `pixel_values.shape[0]` in the supported Qwen image layout
`(num_patches, patch_dim)`: patches **after image preprocessing and before
spatial merging**, not original file bytes, original image pixels, or merged
embedding tokens. An image exactly at the threshold still uses similarity
matching. Videos and unavailable or unsupported image tensors retain their
existing behavior. Configure the same threshold on every DP for consistent
policy; this local admission setting does not change shared score compatibility.

Exact embedding hits still take priority. On an exact miss, a small image skips
both similarity prewarming and final matching, including cached similarity
matches. Its encoder runs normally and the resulting embedding is saved under
the original hash. The gate does not compute/register pHashes or persist resized
tensors for skipped images. These decisions count as `misses` in the existing
statistics, including the denominator of `similarity_lookup_hit_rate`; they do
not represent attempted similarity lookups. Existing statistics reporting RPCs
and exact-cache probes still run.

Tune the threshold using encoder latency, scheduler latency and end-to-end
throughput on the actual model/NPU workload. Processor resizing can enlarge a
small original image. The relative cost also depends on batch size, candidate
count, cache state and Memcache/index latency.

## Optional independently managed mode

For an index whose lifecycle is independent of DP0, set
`"shared_index_mode": "external"` on every scheduler and start one service in the
normal vLLM environment:

```bash
python -m vllm_ascend.distributed.ec_transfer.similarity_index_service \
    --endpoint tcp://127.0.0.1:5558 \
    --namespace qwen-vl-deployment
```

In this mode, all schedulers only connect; none starts or stops the service.
The service's namespace must match `shared_index_namespace`. Stop it separately
after the schedulers have stopped. Do not run an external server on an endpoint
that an embedded DP0 is configured to bind.

## Compatibility, capacity, and failure behavior

One service serves one namespace. The first client establishes the model ID,
vision merge size, pHash/SSIM algorithm version, band count, Hamming limit, SSIM
threshold/range, and capacity settings. Every subsequent request validates these
settings, so incompatible schedulers fail explicitly rather than sharing scores.
Use a separate service and namespace for each deployment with different model
weights, model revisions, processor configuration, or embedding layout. The model
ID alone does not fingerprint the contents of a local model directory. This
feature does not change Memcache embedding keys or isolate different models'
embedding stores.

`max_capacity`, `eviction_threshold`, and `lower_watermark` now apply to the
shared index as a whole. Registration, bidirectional score updates, and eviction
are atomic. Removing metadata updates the reverse and band indexes together;
it does not delete Memcache data.

Metadata may be visible before a resized tensor or embedding is ready. Readers
check data availability and fall back to normal encoder computation when needed.
Cached SSIM results also verify embedding availability and skip the query itself.
Concurrent first encounters may still both miss while an embedding is being
computed; the shared index is not an embedding-publication barrier.

A transport timeout discards the socket and starts a cooldown. During an outage,
similarity lookups become misses and the worker still computes/saves embeddings.
An operation can wait up to twice the configured timeout (send plus receive).
The next call after cooldown reconnects. Namespace, model, and configuration
mismatches are errors, not outage fallbacks.

The service is in-memory. Restarting it clears the index, while Memcache data
remains. There is no persistent index recovery or scan of existing Memcache keys;
candidates are learned again from images that enter similarity processing.
Clients do not retain negative lookup results or old remote index snapshots.

## Shared hit statistics

With a shared index configured, schedulers also share cumulative cache-decision
statistics through that same endpoint. No additional service or configuration is
needed. Each scheduler retains its local counters and publishes one cumulative
snapshot after each batch containing cache loads or saves, and once at shutdown.
Empty/decode-only batches do not add statistics RPCs. Reporting failure does not
change cache actions or fail inference; the next active batch or shutdown retries
the full local counters. There is no periodic retry thread.

Counters are incremented in `update_state_after_alloc` for each multimodal
feature whose final cache action is resolved:

- `exact_hits`: reuse the embedding under the original image hash.
- `similarity_hits`: reuse another image's embedding after a cached or newly
  computed SSIM match and an embedding-availability check.
- `misses`: schedule encoder computation and a save, including unavailable-index
  and missing-candidate fallbacks.

These are scheduler decisions, not request counts, globally unique image hashes,
or successful worker loads. Batch load/save sets deduplicate keys, but decision
counters do not. Features served without entering the connector's allocation
callback are outside this denominator. If an embedding disappears before the
worker loads it, the scheduler's hit count is not decremented.

Let E, S and M be the sums of all reported exact hits, similarity hits and misses,
and T = E + S + M. DP=1 and multi-DP use the same formulas:

| Log field | Formula | Meaning |
| --- | --- | --- |
| `hit_rate` | (E + S) / T | Overall connector cache hit rate. |
| `similarity_hit_rate` | S / T | Fraction of all decisions satisfied by similarity. |
| `similarity_lookup_hit_rate` | S / (S + M) | Similarity hit rate after exact-cache misses. |

An empty denominator gives zero. Sum counters before calculating rates; do not
average the DP percentages. For example, DP0 reporting (8, 1, 1) and DP1 reporting
(0, 2, 18) produce (8, 3, 19), with overall hit rate 36.67%, similarity hit rate
10.00%, and similarity lookup hit rate 13.64%.

Existing batch DEBUG logs are labeled `scope=local`. Additional DEBUG lines
labeled `EC cache stats: scope=global` contain shared counts, all three rates,
`epoch`, and `reporting_instances`. Shutdown attempts a final report and logs its
global snapshot at INFO. The connector's Python `hit_rate` property remains
local; `cache_statistics` exposes the local immutable counts and rates.

A global line is already aggregated: **do not sum global lines across DPs or
across time**. It contains the latest received report per scheduler instance,
not a synchronized barrier. After a completed benchmark, use the latest global
snapshot after every DP has finished reporting. A shutdown line alone does not
prove other DPs have finished. Missing reports or a reporting outage can make a
snapshot incomplete; local counters remain available for reconciliation.

Each scheduler instance gets a new ID at construction. Repeated or older
snapshots cannot inflate or reduce its counters. The index retains counts from
exited instances; `reporting_instances` is the number of instances seen in this
statistics epoch, including exited ones, **not the current DP size**. Metadata
eviction does not reset counters. An index restart starts a new `epoch`: live
schedulers restore their full local cumulative counts on their next report, but
counts from already stopped schedulers are lost. Consequently, an epoch's totals
are not necessarily counts collected only since that epoch began. The statistics
are in memory, not durable accounting.

For DP comparisons, use a fresh deployment per run or subtract before/after
snapshots within one uninterrupted epoch. Keep the dataset, similarity settings,
Memcache/index warmup, and measurement window consistent. Multi-DP concurrency
can still lower hits when similar images arrive before an earlier embedding is
saved; shared counters do not remove that execution-order effect. For a pure
similarity comparison, inspect both S/T and S/(S+M), together with raw counters.

The statistics RPC uses protocol version 2. Restart all schedulers together when
upgrading; older clients/servers are rejected by protocol compatibility checks.

## Focused validation

Run the lightweight CPU index tests without the repository's NPU test bootstrap:

```bash
python -m pytest --noconftest -q tests/ut/distributed/test_similarity_index.py
```

When vLLM is absent, this test module supplies a test-only package bootstrap to
skip runtime logging initialization. Production `vllm_ascend/__init__.py` remains
unchanged; the standalone server CLI is intended for the normal vLLM environment.

With the usual vLLM CPU-unit-test dependencies installed, also run the affected
matcher and connector tests:

```bash
python -m pytest -q tests/ut/distributed/test_perceptual_cache_matcher.py \
    tests/ut/distributed/test_ec_memcache_connector.py
```

Before merging, validate real DP=2 execution: save image A on DP0, wait for its
embedding, send a similar image B with a different original hash to DP1, and
check similarity-hit counters and generated output. Measure scheduler latency
and throughput with and without the shared service on the same NPU workload.
