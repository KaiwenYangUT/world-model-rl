"""meta.json for every run (CONTRIBUTING.md rule 2): git hash, config, hardware, device, wall-clock."""

import datetime
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PACKAGES = ("numpy", "crafter", "gymnasium", "torch", "stable_baselines3", "sb3_contrib",
            "jax", "jaxlib")


def _run(cmd):
    try:
        return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_info():
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"])
    return {"commit": _run(["git", "rev-parse", "HEAD"]),
            "dirty": None if status is None else bool(status)}


def hardware_info():
    info = {"machine": platform.machine(), "platform": platform.platform(),
            "cpu_count": os.cpu_count()}
    if sys.platform == "darwin":
        keys = {"model": "hw.model", "chip": "machdep.cpu.brand_string",
                "perf_cores": "hw.perflevel0.physicalcpu",
                "efficiency_cores": "hw.perflevel1.physicalcpu", "memsize_bytes": "hw.memsize"}
        for name, key in keys.items():
            value = _run(["sysctl", "-n", key])
            info[name] = int(value) if value and value.isdigit() else value
    return info


def package_versions():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


class RunMeta:
    """Collects metadata at start; `finish()` adds wall-clock and writes meta.json."""

    def __init__(self, config, config_path=None, device="cpu"):
        self._t0 = time.perf_counter()
        self.data = {
            "started": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "command": [sys.executable] + sys.argv,
            "config_path": config_path,
            "config": config,
            "git": git_info(),
            "hardware": hardware_info(),
            "device": device,
            "python": platform.python_version(),
            "packages": package_versions(),
            "load_avg_start": list(os.getloadavg()),
        }

    def finish(self, path, **extra):
        self.data["finished"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        self.data["wall_clock_s"] = round(time.perf_counter() - self._t0, 2)
        self.data["load_avg_end"] = list(os.getloadavg())
        self.data.update(extra)
        with open(path, "w") as f:
            json.dump(self.data, f, indent=2, default=str)
            f.write("\n")
        return self.data
