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
        config = json.loads(Path(filename).read_text())
        if config.get('schema') != 'radon_bridge_workspace_v1':
            raise ValueError('Unsupported workspace schema')
        study = config['study']
        if not re.fullmatch(r'\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}', study):
            raise ValueError('Study must be a timestamp without path traversal')
        code, data = Path(config['code_root']), Path(config['data_root'])
        if not code.is_absolute() or not data.is_absolute():
            raise ValueError('Workspace roots must be absolute')
        return cls(code, data, study)

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
