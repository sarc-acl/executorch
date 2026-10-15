#!/usr/bin/env python3
"""Detached, immutable parent cross-build; no deployment or GPU execution."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone


def write_state(path, **values):
    values["updated_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(values, indent=2) + "\n")
    temporary.replace(path)


def hashes(directory):
    return {
        str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob("*"))
        if p.is_file() and not p.is_symlink()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    job = Path(config["job_directory"])
    state = job / "state.json"
    if args.launch:
        job.mkdir(parents=True, exist_ok=False)
        write_state(state, status="queued", phase="starting", config=str(args.config))
        try:
            with (job / "queue.log").open("x") as output:
                subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), str(args.config)],
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        except BaseException as error:
            write_state(state, status="failed", error=str(error))
            raise
        print(state)
        return

    active = None

    def interrupted(signum, frame):
        raise RuntimeError(f"queue interrupted by signal {signum}")

    def run(command, *, env=None, capture=False):
        nonlocal active
        print(json.dumps(command), flush=True)
        active = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE if capture else None,
            text=True,
            start_new_session=True,
        )
        try:
            output, _ = active.communicate()
            rc = active.returncode
        except BaseException:
            os.killpg(active.pid, signal.SIGTERM)
            try:
                active.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid, signal.SIGKILL)
                active.wait()
            raise
        finally:
            active = None
        if rc:
            raise RuntimeError(f"command exited {rc}: {command}")
        return output

    def hold():
        while (Path(config["artifacts"]) / "HOLD").exists():
            write_state(state, status="queued", phase="coordinator_hold", pid=os.getpid())
            time.sleep(30)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        with (job / "queue.lock").open("a") as queue_lock:
            fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            hold()
            repo = Path(config["repository"])
            tools = Path(__file__).resolve().parent
            commit = run(["git", "-C", str(repo), "rev-parse", config["commit"] + "^{commit}"], capture=True).strip()
            control_commit = run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture=True).strip()
            build = Path(config["build_directory"])
            build.mkdir(parents=True, exist_ok=False)
            env = dict(os.environ, WMMA_SOURCE_REPO=str(repo), TMPDIR=config["temporary_directory"])
            Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
            write_state(state, status="running", phase="recursive_export", pid=os.getpid(), commit=commit)
            run(["bash", str(tools / "mktree.sh"), commit, str(build / "source/executorch")], env=env)
            manifest = {
                "source_commit": commit,
                "orchestration_commit": control_commit,
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "recursive_submodules": (build / "source/SUBMODULES").read_text(),
                "export_sha256": hashes(build / "source/executorch"),
                "recipe_sha256": hashes(tools / "jetson-cross"),
                "patches": [],
            }
            (build / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            for directory in (build / "source/executorch").rglob("*"):
                if directory.is_dir() and not directory.is_symlink():
                    directory.chmod(directory.stat().st_mode | 0o200)
            (build / "source/executorch").chmod(0o755)
            lock_path = Path(config["desktop_build_lock"])
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("a") as desktop_lock:
                while True:
                    hold()
                    try:
                        fcntl.flock(desktop_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        write_state(state, status="queued", phase="desktop_build_lock", pid=os.getpid())
                        time.sleep(30)
                hold()
                write_state(state, status="running", phase="compiler_preparation", pid=os.getpid(), commit=commit)
                for key in ("cross_image", "compiler_image"):
                    manifest[key] = json.loads(run(["podman", "image", "inspect", config[key]], capture=True))[0]["Id"]
                if manifest["cross_image"] != config["expected_cross_image_id"]:
                    raise RuntimeError("cross image changed from approved image")
                compiler = build / "compiler"
                compiler.mkdir()
                run([
                    "podman", "run", "--rm", "--userns=keep-id",
                    "-v", f"{compiler}:/compiler:Z", manifest["compiler_image"],
                    "bash", "-euc",
                    "mkdir /compiler/bin /compiler/lib; "
                    "cp /usr/bin/glslc /compiler/bin/glslc.real; "
                    "cp -L /lib64/libSPIRV-Tools.so /lib64/libSPIRV-Tools-opt.so /compiler/lib/",
                ])
                wrapper = compiler / "bin/glslc"
                wrapper.write_text(
                    '#!/bin/bash\ncompiler_root=$(cd "$(dirname "$0")/.." && pwd)\n'
                    'export LD_LIBRARY_PATH="$compiler_root/lib:${LD_LIBRARY_PATH:-}"\n'
                    'exec "$compiler_root/bin/glslc.real" "$@"\n'
                )
                wrapper.chmod(0o755)
                manifest["compiler_sha256"] = hashes(compiler)
                base_command = [
                    "podman", "run", "--rm", "--userns=keep-id",
                    "-v", f"{tools / 'jetson-cross'}:/recipe:ro,Z",
                    "-v", f"{build}:/work:Z", "-v", f"{compiler}:/glslc:ro,Z",
                    "-e", "GLSLC=/glslc/bin/glslc", "-e", "TMPDIR=/work/tmp",
                    manifest["cross_image"],
                ]
                (build / "tmp").mkdir()
                manifest["compiler_version"] = run(base_command + ["/glslc/bin/glslc", "--version"], capture=True)
                (build / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                write_state(state, status="running", phase="cross_build", pid=os.getpid(), commit=commit)
                run(base_command + ["bash", "/recipe/build.sh"], env=env)
                run(base_command + ["bash", "-euc",
                    "cmake -S /recipe/runtime_constants -B /work/metadata-helper -G Ninja "
                    "-DCMAKE_TOOLCHAIN_FILE=/recipe/aarch64.cmake -DCMAKE_BUILD_TYPE=Release "
                    "-DCMAKE_PREFIX_PATH=/work/build "
                    "-DCMAKE_FIND_ROOT_PATH='/work/build;/usr/aarch64-linux-gnu'; "
                    "cmake --build /work/metadata-helper -j8; "
                    "cp /work/metadata-helper/runtime_constants /work/bundle/"], env=env)
                manifest["binary_sha256"] = hashes(build / "bundle")
                modules = list((build / "build").rglob("*.spv"))
                directories = {p.parent for p in modules}
                if len(directories) != 1:
                    raise RuntimeError(f"expected one compiled shader directory, found {directories}")
                spv_directory = directories.pop()
                manifest["spirv_directory"] = str(spv_directory)
                manifest["spirv_sha256"] = hashes(spv_directory)
                (build / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                write_state(state, status="running", phase="golden_check", pid=os.getpid(), commit=commit)
                with (build / "golden.txt").open("x") as output:
                    result = subprocess.run(
                        [sys.executable, str(repo / "sarc/tools/spirv_golden.py"),
                         str(spv_directory), str(build / "source/executorch/sarc/golden/spirv.json")],
                        stdout=output, stderr=subprocess.STDOUT, check=False,
                    )
                manifest["golden_rc"] = result.returncode
                (build / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                if result.returncode:
                    raise RuntimeError(f"parent golden failed; see {build / 'golden.txt'}; no GPU work authorized by this queue")
            write_state(state, status="completed", phase="parent_build_and_golden", commit=commit,
                        build_directory=str(build), manifest=str(build / "manifest.json"), gpu_jobs=0)
    except BaseException as error:
        traceback.print_exc()
        write_state(state, status="failed", error=str(error), pid=os.getpid(), gpu_jobs=0)
        sys.exit(1)


if __name__ == "__main__":
    main()
