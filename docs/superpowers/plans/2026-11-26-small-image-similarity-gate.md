# Small-image similarity gate implementation plan

> Execute the approved design inline using test-driven development and review.

**Goal:** Avoid optional similarity work for small preprocessed images.

**Architecture:** Add validated configuration and one connector predicate used
before prewarming and after exact-hit resolution. Keep matcher/index APIs intact.

**Tech stack:** Python dataclasses, PyTorch tensor metadata, pytest.

- [x] Add regression cases in
  `tests/ut/distributed/test_ec_memcache_connector.py` for configuration,
  threshold boundaries, prewarm filtering, exact hits and fallback behavior.
- [x] Run the focused tests and verify failures for the missing feature.
- [x] Add `min_image_patches` to `similarity_cache_config.py`, validate a
  nonnegative integer excluding booleans, and parse the nested setting.
- [x] Add the image-only predicate and both gates to `ec_memcache_connector.py`.
  Return immediately when disabled; record small-image misses without feature
  data so no similarity persistence occurs.
- [x] Document units, default, strict boundary, statistics and tuning in
  `docs/source/developer_guide/shared_similarity_index.md`; extend the existing
  pending release fragment.
- [x] Run focused connector/matcher/index tests and `bash format.sh ci`.
  Record environment limitations separately from passing checks.
- [x] Review the diff and address findings.

Delivery: commit with sign-off and push to `origin` branch
`codex/shared-similarity-index` without force.

## Validation evidence

- The 20 new cases failed before implementation and passed afterward.
- Connector and matcher suites: 130 passed with CPU PyTorch 2.15.0 and pytest
  9.2.0 on Windows/Python 3.14. A local dependency-isolation harness substituted
  unavailable vLLM base/logger/parallel imports and the native Memcache backend;
  connector, matcher, configuration, index and tensor operations were real.
  This does not replace validation with the supported vLLM/NPU environment.
- Standalone index suite: 19 passed with
  `python -m pytest --noconftest -q tests/ut/distributed/test_similarity_index.py`.
- The original commit reproduced seven existing connector/matcher test failures.
  Fixed missing request mock fields, an incorrect persistence-order assertion,
  and a tensor-dictionary comparison. No corresponding production changes were
  needed. The order assertion now runs outside the caught persistence callback.
- Changed Python files passed Ruff lint and formatting checks. Changed Markdown
  passed PyMarkdown with repository settings. Changed files passed Codespell;
  release-note YAML parsed successfully and `git diff --check` passed.
- `bash format.sh ci` was attempted after installing pre-commit. Its hook setup
  failed fetching `https://github.com/codespell-project/codespell/` because the
  connection to GitHub port 443 failed. The full repository check remains
  outstanding; the targeted checks above are not a substitute for all hooks.
- NPU inference, DP end-to-end validation and threshold performance tuning
  remain outstanding. No nonzero threshold is enabled by default.
