# Development

Create and activate a dedicated Python 3.11 virtual environment, then run
`bash scripts/bootstrap.sh`, `python scripts/manage.py check`,
`python scripts/manage.py test`, and `python scripts/manage.py preflight --machine ibex`.
Preflight checks structure and paths only; it does not certify data, GPU capacity or training.
Integration scripts live under tests/integration and require explicit invocation.

Run CPU acceptance in a clean checkout with only the declared dependencies and
the pinned framework installed. Do not add a sibling Model_Training checkout to
PYTHONPATH: that can hide undeclared imports. Evaluation and diagnostics own their
metrics inside this application. Real-data study entry points that explicitly
load a pinned training snapshot remain a separate integration boundary.

A successful local test run is not a successful GitHub Actions run. Check the
Actions result for the exact pushed SHA before calling a revision accepted.
The required `Research checks` workflow covers clean installation, framework
identity, the complete unit suite and the two-rank MHD trainer reference.
Dataset/parent replay, full-input GPU resource and recovery checks, and scientific
completion have separate receipts; none is implied by a green CPU workflow.

See ../workspace/REPOSITORY_STANDARD.md for new-project reuse and archival rules.
Install the fixed MHD checkout separately; do not bundle framework classes in this wheel.

`python scripts/manage.py verify-env` verifies actual imported framework bytes against the lock and editable application origin.

## Internal architecture

Follow [architecture.md](architecture.md). Keep the source and test role directories identical to the peer application. Run `python scripts/manage.py check` before committing; this checks real modules and import targets, not just top-level folders. Run integration scripts as `python -m tests.integration.<module>` from the repository root. New method-specific files belong under the matching role.

## GPU execution

See [GPU development and batch execution](gpu_workflow.md): ws02 debugging, Ibex A100 preflight and finite batches within one allocation, with project-independent artifacts.
