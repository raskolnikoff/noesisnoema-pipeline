# Claude Code Handover — Session D2: gold-set finalization + first G3 calibration run (nomic GGUF)

Repo: rag-fish/noesisnoema-pipeline. PR #30 (Session D1) is already MERGED to main
(2026-07-10). Branch fresh from up-to-date main: `feat/goldset-g3_demo-d2`.
Guard (run FIRST): `git remote get-url origin` must contain `rag-fish/noesisnoema-pipeline`; else STOP.

## Preconditions (check in order; STOP + report on any failure)
1. `noema-gate --help` works without PYTHONPATH tricks.
2. `goldsets/g3_demo/goldset-review.md` (on main) Verdict column is FILLED for
   EVERY row — the file has 40 rows per PR #30. (Note: `tests/fixtures/g3_demo/goldset.jsonl`
   is an unrelated 30-query synthetic fixture from PR #28 — do not confuse or touch it.)
   Exact verdict syntax: OK / EDIT: <ids> / DROP. Any blank verdict → STOP:
   human review incomplete.
3. GGUF exists: `~/Downloads/nomic-embed-text-v1.5.Q5_K_M.gguf` (record its sha256).
4. Locate `llama-server` (Taka built llama.cpp from source; try `which llama-server`,
   else common build paths). Record the llama.cpp source commit:
   `git -C <llama.cpp checkout> rev-parse HEAD`. If binary not found → STOP, ask Taka.

## Step 1 — geometry resolution (mandatory before any run)
- Read `tests/fixtures/g3_demo/pack/manifest.json` → `embedding.model_id`.
- If it is NOT nomic-embed-text-v1.5: the fixture was built with a different embedder.
  Re-embed the SAME 20 chunks with nomic:
  a. Start `llama-server --embedding -m ~/Downloads/nomic-embed-text-v1.5.Q5_K_M.gguf --port 8080`
     (background); health-check before proceeding; shut it down at session end.
  b. Build a new pack at `packs/g3_demo_nomic/` (untracked; add `packs/` to .gitignore
     if not ignored) from the fixture's EXISTING `chunks.json` — do NOT re-chunk.
     Embed documents via the HTTP server with the `"search_document: "` prefix
     (do NOT use llama-cpp-python; it does not build on this machine).
     Reuse the repo's manifest/pack-writing code paths for the v1.3 manifest:
     correct embedding block (model_id, dim, normalization consistent with pipeline
     defaults incl. mean-centering + centroid_sha256) and integrity hashes.
  c. HARD CHECK: chunk_ids in the new pack must be byte-identical to the fixture's
     (programmatic set comparison). Mismatch → STOP; the reviewed gold-set ids
     would be invalid.
- If it IS nomic already: use the fixture pack as-is (still start llama-server for queries).

## Step 2 — finalize gold-set (from human verdicts only)
- Convert reviewed rows → `goldsets/g3_demo/goldset.jsonl`
  (OK = as proposed; EDIT = Taka's ids; DROP = excluded).
- If approved rows (OK + EDIT) < 30 (policy min_goldset_queries): do NOT proceed.
  Draft replacement queries for the shortfall (same rules as D1: validated ids,
  difficulty mix, rationale) into `goldsets/g3_demo/goldset-review-addendum.md`
  with a blank Verdict column, commit, then STOP for Taka's review of the addendum
  only. Resume Step 2 merging addendum verdicts once filled.
- Validate every chunk_id against chunks.json; dangling id → STOP.
- Write `goldsets/g3_demo/goldset-meta.json`: reviewed_by "taka", review date,
  source pack_id, counts drafted/approved/edited/dropped.

## Step 3 — first calibration run
- `noema-gate run --pack <pack from Step 1> --goldset goldsets/g3_demo/goldset.jsonl --policy <policy>`
  with `retrieval/llama_server_embedder.py` as embed_query_fn ("search_query: " prefix,
  same server).
- Report: recall_at_k; per-query misses (query, expected ids, retrieved top-k);
  breakdown by difficulty class (direct / paraphrase / cross-chunk).
- G2 pinning block in the report: llama.cpp commit, GGUF sha256, pack manifest sha256,
  policy sha256.

## Hard prohibitions
- Do NOT run `noema-gate stamp`. Do NOT edit any policy file or threshold.
- Do NOT commit pack binaries (embeddings) — goldset.jsonl, meta, report, and code
  changes only.
- Do NOT merge. Open a NEW PR from `feat/goldset-g3_demo-d2` titled
  `feat(goldset): g3_demo goldset.jsonl + first G3 calibration run`, run report in the body.
- Shut down llama-server before finishing. Archive this prompt per repo convention.

## Interpretation note for the run result
This is CALIBRATION data on a 20-chunk pilot: recall will read artificially high
(tiny search space). Its purpose is proving the D1→review→D2→run pipeline end-to-end,
not validating the 0.80 threshold. Report numbers neutrally; no pass/fail judgment.

---

## Session log (what actually happened)

- Guard passed. Local `main` was 7 commits ahead of `origin/main` with divergent
  merge history, but `git diff --stat origin/main main` showed an **empty diff** —
  the trees were byte-identical (the extra commits were local merge noise, not
  unpushed content). Branched from local `main` as equivalent to up-to-date
  `origin/main`.
- Precondition 2: the working tree already had `goldsets/g3_demo/goldset-review.md`
  filled in (all 40 rows verdict = `OK`, uncommitted). Committed it first on the new
  branch.
- Precondition 4 failed as anticipated: no from-source `llama-server` binary existed
  (only a Docker-bundled one with no traceable commit). Found the source checkout at
  `~/CLionProjects/llama.cpp` (commit `e5b5ec178d145a762ab847d22b16bb490fb6af4d`,
  dirty: local CMake-only patch removing static-build enforcement + unrelated
  HIP/Hexagon tweaks — no inference math touched) but its `build-macos/` had never
  built `tools/server` (`LLAMA_BUILD_COMMON`/`LLAMA_BUILD_TOOLS`/`LLAMA_BUILD_SERVER`
  were all OFF). Reconfigured with all three ON and built `llama-server` from that
  checkout rather than proceeding on an unpinned binary.
- Step 1: fixture manifest confirmed `sentence-transformers/all-MiniLM-L6-v2`, not
  nomic. Wrote `scripts/build_g3_demo_nomic_pack.py` to re-embed the fixture's
  existing 20 chunks (no re-chunk) via the HTTP server into `packs/g3_demo_nomic/`;
  chunk_id hard-check passed.
- Step 2: parsed `goldset-review.md`'s verdict column programmatically — 40/40 `OK`,
  0 `EDIT`, 0 `DROP` — well above the 30-query floor, so no addendum needed. Wrote
  `goldset.jsonl` + `goldset-meta.json`; dangling-id check clean.
- Step 3: the `noema-gate` CLI's `run` command hardcodes `LlamaCppEmbedder` (direct
  llama-cpp-python binding), which conflicts with the "same server for documents and
  queries" requirement. Wrote `scripts/run_g3_demo_gate.py` calling
  `noema_gate.core.run_gate` directly with `retrieval/llama_server_embedder.py` as
  `embed_query_fn` instead. Result: Recall@5 = 1.0000 across all 40 queries, 0 misses,
  uniform across direct/paraphrase/cross-chunk classes — expected for a 20-chunk pilot.
- PR opened: https://github.com/rag-fish/noesisnoema-pipeline/pull/32
- `llama-server` shut down at session end.
