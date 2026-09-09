"""Machine paths for new studies, independent of model topology and data splits."""
from dataclasses import dataclass
from pathlib import Path
import json
import re

@dataclass(frozen=True)
class Workspace:
    code_root: Path
    data_root: Path
    study: str
    project: str = 'Radon_Bridge'

    @classmethod
    def load(cls, filename):
        filename = Path(filename).resolve()
        config = json.loads(filename.read_text())
        if config.get('schema') not in ('radon_bridge_workspace_v1', 'research_deployment_v1'):
            raise ValueError('Unsupported workspace schema')
        study = config['study']
        if not re.fullmatch(r'\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}', study):
            raise ValueError('Study must be a timestamp without path traversal')
        root = next((p for p in filename.parents if (p / 'project.json').is_file()), filename.parent)
        def anchored(value):
            path = Path(value).expanduser()
            return (path if path.is_absolute() else root / path).resolve()
        project = config.get('project', 'Radon_Bridge')
        if not re.fullmatch(r'[A-Za-z0-9_]+', project):
            raise ValueError('Invalid project name')
        return cls(anchored(config['code_root']), anchored(config['data_root']), study, project)

    @property
    def code(self):
        return self.code_root / self.project / self.study

    @property
    def run(self):
        return self.data_root / self.project / 'runs' / self.study

    @property
    def ophthalmology(self):
        return self.data_root / 'UKBiobank' / 'ophthalmology'

    def describe(self):
        return {k: str(v) for k, v in dict(project=self.project, study=self.study,
                code=self.code, run=self.run, ophthalmology=self.ophthalmology).items()}
