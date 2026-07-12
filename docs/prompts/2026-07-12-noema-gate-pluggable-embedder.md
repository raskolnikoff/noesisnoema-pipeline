Repo: rag-fish/noesisnoema-pipeline (feature branch + PR)
Guard: git remote get-url origin must contain rag-fish/noesisnoema-pipeline; else STOP.

Task: make the embedder pluggable in the noema-gate CLI.
1. Add `--embedder {llama-cpp|llama-server}` and `--server-url` options to
   `noema-gate run`, wiring retrieval/llama_server_embedder.py for the
   llama-server case. Default preserves current behavior.
2. Fold scripts/run_g3_demo_gate.py's logic into this path, then DELETE the
   script (single audited execution path for the gate).
3. Verify and test that a gate run via the CLI emits the expected audit
   events (gate.run) to the chain — add a test asserting event emission.
4. While there: reconcile the policy threshold question — report which policy
   file the g3_demo run consumed, its sha256, and whether its g3 threshold
   matches the canonical RAGfish policy (0.80). Do NOT change any policy
   value; report only.
Tests green, PR only, do not merge. Archive prompt per convention.

---

## Session log (what actually happened)

- Guard passed. Branched fresh from `origin/main` as
  `feat/noema-gate-pluggable-embedder` (independent of the still-open
  `feat/goldset-g3_demo-d2` PR #32 from the prior session — local `main`'s
  content was confirmed identical to `origin/main` again, same as last time).
- `noema_gate/cli.py`'s `run` command gained `--embedder {llama-cpp|llama-server}`
  (default `llama-cpp`, behavior-preserving), `--server-url` (defaults to
  `retrieval.llama_server_embedder.DEFAULT_SERVER_URL`), and an optional
  (server-mode only) `--gguf` used solely to hash the GGUF file for the audit
  event's `model_id` — the server has no model-identity endpoint over HTTP, so
  without `--gguf` that field is `""`. Added `_check_server_health` so an
  unreachable server fails fast with a clear message instead of a deep
  exception inside `RetrievalHarness`.
- `scripts/run_g3_demo_gate.py` does not exist on this branch — it was only
  ever committed on `feat/goldset-g3_demo-d2` (Session D2, PR #32), which is
  still unmerged. Nothing to delete here; the CLI change is the "single
  audited execution path" the script's logic now folds into once that PR
  lands (or is superseded).
- Added `tests/noema_gate/test_cli_run_embedder.py`: exercises `noema-gate run
  --embedder llama-server` end-to-end via `typer.testing.CliRunner`
  (`_check_server_health` and `make_embed_query_fn` monkeypatched — no real
  server or GGUF needed, matching the existing no-GGUF-required test
  convention). Asserts a `gate.run` event lands in the audit log with the
  expected `action`/`actor`/`plane`/`triplet` fields and valid hash-chain
  linkage (`prev_hash` = genesis, non-empty `event_hash`). Also covers:
  `--gguf` present (model_id = real file sha256), an unknown `--embedder`
  value (clean exit 1), and an unreachable server (clean exit 1, no hang).
- Full suite: `355 passed, 21 skipped` (skips are pre-existing GGUF/network-
  gated tests, unrelated to this change).
- Policy reconciliation (report only, nothing changed):
  - The Session D2 g3_demo calibration run consumed
    `tests/fixtures/g3_demo/policy.yaml` — sha256
    `077b4d43de227775d5f84d6c5d85432c421ee53d5fbdbf27d2c219e94657463f` —
    `gates.g3_promotion.threshold: "0.60"`.
  - Canonical policy (`rag-fish/RAGfish`, `policy/noema-policy-v0.8.yaml`,
    fetched read-only via `gh api`, git blob sha
    `3ba28021ece300ba76838b9a3ec0e01857abec23` independently matches the
    downloaded bytes) — sha256
    `673b287d5a0fb9fb6d72ba3aac539dee6d26274f882a75df75bc08e6a61ccc67` —
    `gates.g3_promotion.threshold: "0.80"`.
  - **They do not match** — by design, not drift: `scripts/build_g3_demo.py`
    deliberately calibrates the demo fixture's policy to `0.60` because the
    fixture's checked-in pack uses `sentence-transformers/all-MiniLM-L6-v2`
    (a weaker, 384-dim embedder), not the production `nomic-embed-text-v1.5`
    — "this is a mechanism demo, not a production calibration" (per that
    script's own docstring). The D2 session's actual nomic re-embedding run
    (Recall@5 = 1.0000) would also clear the real 0.80 threshold, but it was
    evaluated against the lenient demo policy, not the canonical one — no
    pack has yet been gated against `policy/noema-policy-v0.8.yaml` itself.
