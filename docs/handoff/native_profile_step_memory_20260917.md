# Native profile reuse: worker step memory gate (2026-09-17)

Prepared on research branch `2026_09_17_12_46_00` after `fc58148`.
Implemented and isolated CPU-tested; not deployed, not GPU/runtime accepted.

## Defect and bounded correction

`project_dispatch.py` requests a 128 GiB allocation but starts its native worker
with `--mem=100G`. Requalification previously checked only allocation RAM. A
previous full-cohort peak of 105 GiB, a new short-probe peak of 50 GiB and a 2 GiB
owner passed 128 * 0.85 while exceeding the actual worker step's 100 GiB limit.

`live_envelope` now requires the current RUNNING step's Slurm TRES memory as
`worker_ram_gib`, separately from allocation RAM. `qualify` requires both:

- max(previous full peak, current peak) <= worker step memory limit;
- other RAM + that maximum <= 0.85 * allocation RAM.

There is no additional 15% margin on the worker step. Both observed limits are
stored in the requalification receipt. Missing, non-finite, non-positive or
inconsistent step limits refuse reuse; unknown/non-numeric resource evidence
also refuses qualification. New callers must provide the step limit; there is
no fallback to the allocation value. Scientific recipes, batches, quotas,
checkpoint/spec identity, full-probe coverage and pinned execution sources are
unchanged. The existing dynamic `**live_envelope(...)` caller carries the new
required field without dispatch changes.

## Evidence and limits

- Independent Ibex CPU snapshot `/tmp/radon_step_budget_20260917_mzJMmd`:
  `tests/unit/runtime/test_native_profile_reuse.py`: **44 passed in 0.12s**.
- Cases include 105/50/100 refusal, 90/50/100 acceptance, 100/50/100 boundary
  acceptance, current peak 101/step100 refusal, unknown Slurm step memory and
  non-finite/non-positive/boolean measured evidence. Existing allocation-total
  rejection remains covered.
- Local `python3 scripts/manage.py check` and `git diff --check`: passed.
- Local default Python has no pytest. First isolated remote attempt omitted
  package `__init__.py` files and failed collection; subsequent self-contained
  snapshots include them. No production import fallback was used.
- Final source SHA256: `59726974286faa4478f72f7828d2fa6235352b955c5164650fa4e7522df4439a`.
- Final test SHA256: `1ad064a462ac1feba2309efd37ea1d97750708c5d2cf39284cb4e642bb84a25f`.

Only an independent `/tmp` CPU snapshot was copied/tested on Ibex. No production
files, workers, allocation owners, pending pins or GPU submissions were changed.
No real GPU acceptance or speed improvement is claimed. Root integration owner
must review and integrate the research commit, then separately validate a future
immutable worker snapshot before deployment. Healthy existing workers retain
original source. The production incident is not closed by these CPU tests.
