# Small-image similarity gate

The approved design adds `similarity_cache.min_image_patches`, a nonnegative
integer defaulting to zero (disabled). For the supported Qwen image layout,
`pixel_values.shape[0]` measures patches after preprocessing, before merging.
Images strictly below the threshold skip similarity prewarming and final
matching. Exact embedding hits retain priority. Misses still compute and save
embeddings, but do not persist resized tensors or register similarity metadata.
Videos and unavailable or unsupported image tensors retain existing behavior.

The connector owns the policy; the matcher and shared-index protocol are
unchanged. Every DP should use the same threshold. The threshold does not alter
score compatibility. Shape inspection avoids tensor reads and device syncs.
Skipped images count as misses in the existing statistics denominator.

Tests cover validation, default behavior, strict threshold boundaries, mixed
prewarm batches, all-small batches, exact hits, missing data, videos, and final
decisions without prewarming. NPU benchmarking remains necessary to select a
profitable threshold; no performance claim or nonzero default is introduced.
