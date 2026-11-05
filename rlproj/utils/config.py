"""Load experiment configs (one YAML file per experiment, CONTRIBUTING.md rule 1).

Uses ruamel.yaml, which crafter already depends on, so it is present in both venvs.
"""

from ruamel.yaml import YAML


def load_config(path):
    with open(path) as f:
        cfg = YAML(typ="safe").load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return cfg
