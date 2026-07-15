Claude Code Handover — G3 gate report preservation (2 sessions, 2 repos)

Background: PR #32's gate run report (report-detail.json) was never committed and is
gone from disk. noema-gate stamp writes eval_report_sha256 into the manifest —
a hash pointing at a lost file is a dangling audit reference. Reports must be durable
BEFORE the first production stamp.

SESSION E2 — implementation (rag-fish/noesisnoema-pipeline, feature branch + PR)

Guard (FIRST): git remote get-url origin must contain rag-fish/noesisnoema-pipeline;
else STOP. Branch from up-to-date main: feat/g3-report-preservation.

Implement the E1 amendment exactly:

noema-gate run: default report path reports/g3/<pack_id>/<UTC-ts>-report.json
(create dirs); refuse to overwrite an existing file. Keep --out override.
noema-gate stamp: before writing the promotion block, verify the report file
is git-tracked (git ls-files --error-unmatch <path> semantics via subprocess
or dulwich-free equivalent); refuse if untracked, unless
--allow-untracked-report, which emits a distinct audit event
(gate.stamp with detail.override = "untracked_report").
Confirm gate.run audit event includes the report sha256 in outputs.sha256_refs
(add if missing).
Ensure reports/ is NOT excluded by .gitignore; add reports/g3/.gitkeep.
Tests: default-path creation; overwrite refusal; stamp-refusal on untracked
report (tmp git repo fixture); override event emission; report-hash presence
in the audit event.
Retroactive note (no fabrication): do NOT attempt to reconstruct PR #32's lost
report. Add a one-line note to goldsets/g3_demo/goldset-meta.json:
"report_status": "lost_pre_preservation_policy" — honest record beats
backfilled evidence.

Constraints: English comments; tests green; PR only, do not merge.
PR title: feat(gate): report preservation — durable eval evidence for stamp (spec 2026-07-14).
Archive prompt per convention.

---

## Session log (what actually happened)

- Guard passed. Branched from local `main` (which already contained everything
  on `origin/main` plus 8 unpushed local commits — no divergence) as
  `feat/g3-report-preservation`.
- **Premise check before acting on the "retroactive note" step**: the handover's
  background claims PR #32's `report-detail.json` "was never committed and is
  gone from disk." This was checked and found false — `goldsets/g3_demo/report.json`
  and `goldsets/g3_demo/report-detail.json` are both present on disk *and*
  committed at `14ead91` (`feat(goldset): g3_demo goldset.jsonl + first G3
  calibration run (#32)`), which is merged into `origin/main` (`gh pr view 32`
  confirms `state: MERGED`). Writing `"report_status":
  "lost_pre_preservation_policy"` into `goldset-meta.json` would itself have
  been a fabricated audit claim, directly contradicting the same instruction's
  "no fabrication... honest record beats backfilled evidence" principle. Flagged
  this to the user before proceeding; decision: skip the note entirely rather
  than write an inaccurate one. `goldsets/g3_demo/goldset-meta.json` is
  unchanged by this session.
- The rest of the spec stands on its own engineering merit regardless of the
  stale premise, and was implemented as specified:
  - `noema_gate/core.py::run_gate`: default `out_path` (when `--out`/`out_path`
    is not given) is now `reports/g3/<pack_id>/<UTC-ts>-report.json`
    (`_default_report_path`), parent dirs created via `mkdir(parents=True,
    exist_ok=True)`. An existing file at the resolved path (default *or*
    explicit `--out`) raises `GateRefusalError(exit_code=2)` rather than being
    silently overwritten — report evidence is write-once. `out_path`
    resolution moved after `pack = load_pack(pack_dir)` since the default
    needs `pack.manifest["pack_id"]`.
  - `noema_gate/core.py::stamp_gate`: new `_is_git_tracked()` helper shells out
    to `git -C <dir> ls-files --error-unmatch -- <abs_path>` (exit 0 = tracked;
    any failure — including git binary missing or path outside a repo — is
    treated as untracked, the conservative default). `stamp_gate` now checks
    this before touching the promotion block and refuses
    (`GateRefusalError`) unless the new `allow_untracked_report=True` param is
    passed. `StampResult` gained `untracked_report_override: bool` so the CLI
    layer knows whether to audit the override.
  - `noema_gate/cli.py`: added `stamp --allow-untracked-report`. When the
    override fires, `detail["override"] = "untracked_report"` is added to
    whichever `gate.stamp`/`gate.stamp.forced` audit event is emitted (both
    branches, so `--force` + `--allow-untracked-report` together still audit
    correctly).
  - `gate.run`'s audit event already carried `outputs.sha256_refs =
    [report_sha256(result.report_path)]` from the prior pluggable-embedder
    session — confirmed unaffected by the report-path refactor and pinned
    with an explicit new test (no code change needed here).
  - `.gitignore`: no existing pattern actually excluded `reports/g3/*.json`
    (the broad `*.jsonl`/`*.npy` ignores don't match `.json`), but added an
    explicit `!reports/` / `!reports/g3/` / `!reports/g3/**` allow-list block
    (matching the repo's existing `tests/fixtures/g3_demo/` allow-list
    convention) so a future broader ignore rule can't silently swallow report
    evidence. Added `reports/g3/.gitkeep`.
- **Existing-test fallout from the new default report path**: most
  `tests/noema_gate/` call sites invoked `run_gate`/`stamp_gate` without
  `out_path`/`allow_untracked_report`, which under the new defaults either (a)
  collided on the same-second `reports/g3/pack-demo/<ts>-report.json` filename
  across the many tests sharing `pack_id="pack-demo"`, polluting the real repo
  tree, or (b) correctly refused to stamp a report under `tmp_path` (not a git
  repo — genuinely untracked). Fixed by giving every pre-existing test an
  explicit isolated `out_path` (`pack_dir / "report.json"`, restoring the old
  per-test isolation) and `allow_untracked_report=True` where the test isn't
  specifically about report preservation (`test_g3_gate.py`,
  `test_g3_demo_fixture.py`, `test_cli_run_embedder.py`). Verified no stray
  files were left in the real `reports/` tree afterward.
- New `tests/noema_gate/test_report_preservation.py` (8 tests): default-path
  creation + dir creation; overwrite refusal; stamp refusal on an untracked
  report using a real throwaway `git init` fixture (`git_repo`); stamp success
  once the report is `git add`+committed; `allow_untracked_report=True`
  override at the `core` layer; CLI-level override emitting the
  `detail.override = "untracked_report"` audit event; CLI-level refusal
  without the override flag; `gate.run` audit event's `outputs.sha256_refs`
  pinned against `report_sha256()`.
- Full suite: `363 passed, 21 skipped` (skips pre-existing, GGUF/network-gated,
  unrelated to this change).
- PR opened (not merged): branch `feat/g3-report-preservation`.
