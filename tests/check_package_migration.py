"""Namespace identity, historical pickle imports, and safe workspace configuration."""
import importlib
import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

def check():
    for name in ['projector','bridge','graph','model','network_interface','workspace']:
        old = importlib.import_module('radonbridge.' + name)
        new = importlib.import_module('radon_bridge.' + name)
        assert old is new, name
        assert new.__spec__.name == 'radon_bridge.' + name
    from radon_bridge.bridge import FeatureSpec
    original = FeatureSpec.__module__
    try:
        FeatureSpec.__module__ = 'radonbridge.bridge'
        saved = pickle.dumps(FeatureSpec)
    finally:
        FeatureSpec.__module__ = original
    assert pickle.loads(saved) is FeatureSpec
    from radon_bridge.workspace import Workspace
    with tempfile.TemporaryDirectory() as d:
        root = Path(d);p=root/'workspace.json'
        config={'schema':'radon_bridge_workspace_v1','code_root':str(root/'home/mengh'),
                'data_root':str(root/'data/mengh'),'study':'2026_09_09_10_30_34'}
        p.write_text(json.dumps(config));ws=Workspace.load(p)
        ws.code.mkdir(parents=True);ws.run.mkdir(parents=True)
        cmd=[sys.executable,'-m','radon_bridge','--workspace',str(p)]
        report=json.loads(subprocess.check_output(cmd,text=True));assert report['state']=='passed'
        assert report['training_started'] is False and report['labels_accepted'] is False
        result=subprocess.run(cmd+['--require-verified-images'],capture_output=True,text=True)
        assert result.returncode==2 and not json.loads(result.stdout)['checks']['images_verified']
        for wrong in ['../escape','2026_09_09_10_30_34/escape','']:
            config['study']=wrong;p.write_text(json.dumps(config))
            try:Workspace.load(p)
            except ValueError:pass
            else:raise AssertionError('unsafe timestamp accepted')
    subprocess.run([sys.executable,'-m','radonbridge.experiment','--help'],check=True,stdout=subprocess.DEVNULL)
    print(json.dumps({'passed':True,'same_module_and_class_identity':True,'legacy_pickle_loading':True,
                      'legacy_module_cli':True,'unverified_data_rejected':True,'path_traversal_rejected':True}))

if __name__=='__main__':check()
