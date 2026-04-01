import shlex
import subprocess
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path, dry_run: bool = False) -> None:
    print("Running:")
    print(shlex.join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)
