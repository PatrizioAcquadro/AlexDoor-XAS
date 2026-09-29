"""Training lifecycle controls independent of the model and cached observations."""

import json
import math
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class EarlyStopping:
    settings: dict
    reference_score: float | None = None
    bad_evaluations: int = 0

    def __post_init__(self):
        for key in ("patience_evaluations", "min_epochs"):
            if type(self.settings[key]) is not int or self.settings[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not 0 <= self.settings["min_relative_improvement"] < 1:
            raise ValueError("Relative improvement must be in [0, 1)")

    def update(self, score, epoch):
        """Count completed development evaluations, retaining cumulative small gains."""
        if not math.isfinite(score):
            raise FloatingPointError("Nonfinite development selection score")
        threshold = self.settings["min_relative_improvement"]
        if self.reference_score is None or (
            score < self.reference_score and score <= self.reference_score * (1 - threshold)
        ):
            self.reference_score = score
            self.bad_evaluations = 0
        else:
            self.bad_evaluations += 1
        return (
            epoch >= self.settings["min_epochs"]
            and self.bad_evaluations >= self.settings["patience_evaluations"]
        )


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def launch_detached(command, output, cwd, *, resume=False):
    """Submit a single user service; no assistant, polling loop or restart is needed."""
    output = Path(output).resolve()
    if output.exists() != resume:
        raise ValueError(
            "A new run requires an absent output; resume requires its existing directory"
        )
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", output.name):
        raise ValueError("Use a simple run directory name, e.g. run-01")
    output.parent.mkdir(parents=True, exist_ok=True)
    unit = f"alexdoor-perception-{output.name}.service"
    log = output.with_name(output.name + ".console.log")
    subprocess.run(
        [
            "systemd-run",
            "--user",
            "--collect",
            f"--unit={unit}",
            f"--working-directory={Path(cwd).resolve()}",
            "--property=Type=exec",
            "--property=Restart=no",
            f"--property=StandardOutput=append:{log}",
            "--property=StandardError=inherit",
            "--",
            *map(str, command),
        ],
        check=True,
    )
    record = dict(
        submitted_at=datetime.now(UTC).isoformat(),
        unit=unit,
        output=str(output),
        console_log=str(log),
        command=list(map(str, command)),
    )
    with output.with_name(output.name + ".launch.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    return record
