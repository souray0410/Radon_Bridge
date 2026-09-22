"""Read-only deployment preflight; this entry point never launches training."""
import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from radon_bridge.runtime.paths import Workspace

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--require-verified-images', action='store_true')
    p.add_argument('--check-framework', action='store_true')
    args = p.parse_args()
    ws = Workspace.load(args.workspace)
    report = dict(ws.describe(), python=sys.version, training_started=False,
                  gpu_requested=False, labels_accepted=False)
    checks = {'code_exists': ws.code.is_dir(), 'run_exists': ws.run.is_dir(),
              'run_writable': os.access(ws.run, os.W_OK)}
    marker = ws.ophthalmology / 'manifests' / 'verification_status.json'
    report['images'] = json.loads(marker.read_text()) if marker.exists() else {'state': 'not_present'}
    if args.require_verified_images:
        checks['images_verified'] = report['images'].get('state') == 'available_image_files_verified'
    if args.check_framework:
        try:
            import torch
            from mhd_framework.core import MHD_Graph
            from radon_bridge.models.graph import MHDBuilder
            report['torch'] = torch.__version__
            report['torch_cuda_build'] = torch.version.cuda
            report['framework_api'] = 'V5'
            checks['framework_import'] = True
        except Exception as e:
            checks['framework_import'] = False
            report['framework_error'] = repr(e)
    for key, where in [('code_commit', ws.code), ('mhd_commit', ws.code/'third_party/MHD_Framework')]:
        try:
            report[key] = subprocess.check_output(['git','-C',str(where),'rev-parse','HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            report[key] = None
    report['checks'] = checks
    report['state'] = 'passed' if all(checks.values()) else 'not_ready'
    print(json.dumps(report, indent=2))
    return 0 if all(checks.values()) else 2

if __name__ == '__main__':
    raise SystemExit(main())
