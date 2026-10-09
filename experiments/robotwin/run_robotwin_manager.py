import atexit
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SINGLE_ENTRY = PROJECT_ROOT / "experiments" / "robotwin" / "eval_robotwin_single.py"
EVAL_STEP_LIMIT_FILE = PROJECT_ROOT / "third_party" / "RoboTwin" / "task_config" / "_eval_step_limit.yml"
TERMINATE_TIMEOUT_SEC = 10
POLL_INTERVAL_SEC = 2
AUTHKEY_ENV = "FLEXPI_POLICY_AUTHKEY"   # as flexpi_policy/policy_server.py


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    parts = ckpt_path.resolve().parts
    if "runs" in parts:
        runs_idx = parts.index("runs")
        if runs_idx + 2 >= len(parts):
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        task_name = parts[runs_idx + 1]
        date_dir = parts[runs_idx + 2]
        if task_name == "" or date_dir == "":
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        return f"{task_name}_{date_dir}"
    return ckpt_path.stem


def _is_blocked_override(raw_override: str) -> bool:
    key = raw_override.split("=", 1)[0].lstrip("+~")
    if key in {
        "ckpt",
        "gpu_id",
        "EVALUATION.task_name",
        "EVALUATION.task_config",
        "EVALUATION.output_dir",
        "EVALUATION.worker_units_file",
        "EVALUATION.serve_address_file",
        "EVALUATION.policy_server",
    }:
        return True
    return key.startswith("MULTIRUN.") or key.startswith("hydra.")


def _collect_worker_overrides() -> list[str]:
    return [ov for ov in HydraConfig.get().overrides.task if not _is_blocked_override(ov)]


def _load_all_tasks() -> list[str]:
    if not EVAL_STEP_LIMIT_FILE.exists():
        raise FileNotFoundError(f"Task list file not found: {EVAL_STEP_LIMIT_FILE}")
    with EVAL_STEP_LIMIT_FILE.open("r", encoding="utf-8") as f:
        task_map = yaml.safe_load(f)
    if not isinstance(task_map, dict) or len(task_map) == 0:
        raise ValueError(f"Invalid task map in: {EVAL_STEP_LIMIT_FILE}")
    tasks = list(task_map.keys())
    # Keep original order and remove duplicates.
    seen = set()
    dedup_tasks: list[str] = []
    for task in tasks:
        if task in seen:
            continue
        seen.add(task)
        dedup_tasks.append(task)
    return dedup_tasks


def _parse_success_rate(result_file: Path) -> float:
    if not result_file.exists():
        raise FileNotFoundError(f"Result file not found: {result_file}")
    text = result_file.read_text(encoding="utf-8")
    last_value: float | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "":
            continue
        try:
            last_value = float(stripped)
        except ValueError:
            continue
    if last_value is None:
        raise ValueError(f"Failed to parse success rate from: {result_file}")
    return last_value


def _phase_result_filename(phase: str) -> str:
    if phase == "clean":
        return "_result_clean.txt"
    if phase == "random":
        return "_result_random.txt"
    raise ValueError(f"Unsupported phase: {phase}")


def _mean_or_none(values: list[float | None]) -> float | None:
    valid = [v for v in values if v is not None]
    if len(valid) == 0:
        return None
    return float(sum(valid) / len(valid))


def _to_jsonable(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


@dataclass
class RunningState:
    task_name: str
    gpu_id: int
    phase: str  # "clean" | "random"
    process: subprocess.Popen[str]


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_robotwin.yaml")
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None.")
    if not SINGLE_ENTRY.exists():
        raise FileNotFoundError(f"Single evaluation entry not found: {SINGLE_ENTRY}")

    ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt_tag = _resolve_ckpt_tag(ckpt_path)

    robotwin_root = _resolve_path(str(cfg.EVALUATION.robotwin_root), base=PROJECT_ROOT)
    if not robotwin_root.exists():
        raise FileNotFoundError(f"RoboTwin root not found: {robotwin_root}")

    num_gpus = int(cfg.MULTIRUN.num_gpus)
    if num_gpus <= 0:
        raise ValueError("`MULTIRUN.num_gpus` must be > 0.")
    max_tasks_per_gpu = int(cfg.MULTIRUN.max_tasks_per_gpu)
    if max_tasks_per_gpu <= 0:
        raise ValueError("`MULTIRUN.max_tasks_per_gpu` must be > 0.")
    gpu_ids = list(range(num_gpus))

    phases_cfg = cfg.MULTIRUN.get("phases", ["clean", "random"])
    phases: list[str] = [str(p) for p in phases_cfg]
    if len(phases) == 0:
        raise ValueError("`MULTIRUN.phases` must be a non-empty list.")
    if len(phases) != len(set(phases)):
        raise ValueError(f"`MULTIRUN.phases` has duplicates: {phases}")
    for p in phases:
        if p not in ("clean", "random"):
            raise ValueError(f"Unsupported phase in `MULTIRUN.phases`: {p!r}. Expected subset of {{clean, random}}.")

    output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    run_ts = output_dir.name
    if run_ts == "":
        raise ValueError(f"Invalid EVALUATION.output_dir (missing run_ts): {output_dir}")
    run_output_dir = PROJECT_ROOT / "evaluate_results" / "robotwin" / ckpt_tag / run_ts
    run_output_dir.mkdir(parents=True, exist_ok=True)

    manager_log = run_output_dir / "manager.log"
    failed_tasks_file = run_output_dir / "failed_tasks.txt"
    summary_csv = run_output_dir / "summary.csv"
    summary_json = run_output_dir / "summary.json"

    tasks_cfg = cfg.MULTIRUN.get("tasks", None)
    explicit_tasks: list[str] | None = None
    if tasks_cfg is not None:
        explicit_tasks = [str(t) for t in tasks_cfg]
        if len(explicit_tasks) == 0:
            explicit_tasks = None  # treat empty list as unset → all tasks
    task_name_cfg = cfg.EVALUATION.task_name
    if explicit_tasks is not None:
        all_tasks = set(_load_all_tasks())
        unknown = [t for t in explicit_tasks if t not in all_tasks]
        if unknown:
            raise ValueError(f"`MULTIRUN.tasks` contains tasks not in _eval_step_limit.yml: {unknown}")
        tasks = explicit_tasks
    elif task_name_cfg is None or str(task_name_cfg).strip() == "":
        tasks = _load_all_tasks()
    else:
        tasks = [str(task_name_cfg)]

    shard_cfg = cfg.MULTIRUN.get("shard", None)
    shard_str = str(shard_cfg).strip() if shard_cfg is not None else ""
    if shard_str != "":
        try:
            idx_str, count_str = shard_str.split("/")
            shard_index = int(idx_str)
            shard_count = int(count_str)
        except (ValueError, AttributeError):
            raise ValueError(f"`MULTIRUN.shard` must be 'N/K' (N=index, K=count), got: {shard_cfg!r}")
        if shard_count < 1:
            raise ValueError(f"`MULTIRUN.shard` count must be >= 1, got K={shard_count}")
        if not (0 <= shard_index < shard_count):
            raise ValueError(f"`MULTIRUN.shard` index out of range: N={shard_index}, must be 0 <= N < {shard_count}")
        tasks_before = len(tasks)
        chunk = (tasks_before + shard_count - 1) // shard_count  # ceil(tasks_before / shard_count)
        tasks = tasks[shard_index * chunk:(shard_index + 1) * chunk]
        print(f"[manager] shard {shard_index}/{shard_count}: selected {len(tasks)} of {tasks_before} tasks (block)", flush=True)

    extra_overrides = _collect_worker_overrides()

    task_rates: dict[str, dict[str, float | None]] = {
        task: {"clean": None, "random": None} for task in tasks
    }
    failed_records: list[dict[str, Any]] = []
    pending_tasks = deque(tasks)
    running_states: list[RunningState] = []

    phase_to_task_config = {
        "clean": "demo_clean",
        "random": "demo_randomized",
    }

    def _next_phase(p: str) -> str | None:
        idx = phases.index(p)
        return phases[idx + 1] if idx + 1 < len(phases) else None

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with manager_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    def build_cmd(*, task_name: str, gpu_id: int, phase: str) -> list[str]:
        task_config = phase_to_task_config[phase]
        cmd = [
            sys.executable,
            str(SINGLE_ENTRY),
            f"ckpt={str(ckpt_path)}",
            f"gpu_id={gpu_id}",
            f"EVALUATION.task_name={task_name}",
            f"EVALUATION.task_config={task_config}",
            f"EVALUATION.output_dir={str(output_dir)}",
        ]
        cmd.extend(extra_overrides)
        return cmd

    def launch_phase(task_name: str, gpu_id: int, phase: str) -> RunningState:
        cmd = build_cmd(task_name=task_name, gpu_id=gpu_id, phase=phase)
        log(
            f"launch task={task_name} phase={phase} gpu={gpu_id} "
            f"cmd={' '.join(cmd)}"
        )
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            text=True,
        )
        return RunningState(
            task_name=task_name,
            gpu_id=gpu_id,
            phase=phase,
            process=process,
        )

    def terminate_all_running() -> None:
        for state in list(running_states):
            if state.process.poll() is not None:
                continue
            log(f"terminating task={state.task_name} phase={state.phase} gpu={state.gpu_id}")
            state.process.terminate()
        deadline = time.time() + TERMINATE_TIMEOUT_SEC
        for state in list(running_states):
            if state.process.poll() is not None:
                continue
            remaining = max(0.0, deadline - time.time())
            try:
                state.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                log(f"killing task={state.task_name} phase={state.phase} gpu={state.gpu_id}")
                state.process.kill()
                state.process.wait()

    def gpu_running_count(gpu_id: int) -> int:
        count = 0
        for state in running_states:
            if state.gpu_id != gpu_id:
                continue
            if state.process.poll() is None:
                count += 1
        return count

    def _first_incomplete_phase(task_name: str) -> str | None:
        return next((p for p in phases if task_rates[task_name][p] is None), None)

    def try_launch_pending(gpu_id: int) -> None:
        while len(pending_tasks) > 0 and gpu_running_count(gpu_id) < max_tasks_per_gpu:
            task_name = pending_tasks.popleft()
            phase = _first_incomplete_phase(task_name)
            if phase is None:
                continue  # All phases done (resume) — drop and pick another task.
            running_states.append(launch_phase(task_name=task_name, gpu_id=gpu_id, phase=phase))

    def write_outputs() -> None:
        clean_mean = _mean_or_none([task_rates[t]["clean"] for t in tasks])
        random_mean = _mean_or_none([task_rates[t]["random"] for t in tasks])

        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task_name", "clean_success_rate", "random_success_rate"])
            for task in tasks:
                writer.writerow(
                    [
                        task,
                        task_rates[task]["clean"],
                        task_rates[task]["random"],
                    ]
                )
            writer.writerow(["__overall__", clean_mean, random_mean])

        payload = {
            "per_task": [
                {
                    "task_name": task,
                    "clean_success_rate": _to_jsonable(task_rates[task]["clean"]),
                    "random_success_rate": _to_jsonable(task_rates[task]["random"]),
                }
                for task in tasks
            ],
            "overall": {
                "clean_mean_success_rate": _to_jsonable(clean_mean),
                "random_mean_success_rate": _to_jsonable(random_mean),
            },
        }
        summary_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        with failed_tasks_file.open("w", encoding="utf-8") as f:
            for rec in failed_records:
                f.write(
                    f"{rec['task_name']},{rec['phase']},gpu={rec['gpu_id']},"
                    f"return_code={rec['return_code']},reason={rec['reason']}\n"
                )

    log(
        f"manager start tasks={len(tasks)} gpu_ids={gpu_ids} "
        f"max_tasks_per_gpu={max_tasks_per_gpu} phases={phases} output_dir={run_output_dir}"
    )

    # Resume support — pre-populate task_rates from any existing result files
    # under run_output_dir (auto-skip already-finished (task, phase) combos).
    # Triggered just by pointing EVALUATION.output_dir at an existing run dir.
    resumed = 0
    for task in tasks:
        for phase in phases:
            result_file = run_output_dir / task / _phase_result_filename(phase)
            if not result_file.exists():
                continue
            try:
                task_rates[task][phase] = _parse_success_rate(result_file)
                resumed += 1
            except Exception as exc:
                log(f"resume: failed to parse {result_file}: {repr(exc)} — will re-run this phase")
    pending_tasks = deque(
        t for t in tasks if any(task_rates[t][p] is None for p in phases)
    )
    log(f"resume scan: loaded {resumed} existing (task, phase) results; {len(pending_tasks)} of {len(tasks)} tasks pending")

    has_failure = False
    failure_message = ""

    use_server = bool(cfg.MULTIRUN.get("policy_server", False))
    if (use_server or bool(cfg.MULTIRUN.get("persistent_workers", False))) and len(pending_tasks) > 0:
        # Persistent workers: max_tasks_per_gpu processes per GPU, each loading the model once
        # and claiming the next unfinished (task, phase) unit from a shared list until none is
        # left (eval_policy.py: mkdir in .claims is the atomic claim). Results are harvested
        # from the result files as they appear; a failed worker stops all (rerun resumes).
        #
        # Policy server (MULTIRUN.policy_server): one server process per GPU holds the model
        # (serve_robotwin_policy.py), and envs_per_gpu simulator workers per GPU send it their
        # replans. A unit is then one episode (eval_policy.py run_episode_unit), so the workers
        # spread over the tasks' episodes; a task's episodes are merged into its _result file.
        n_episodes = int(cfg.EVALUATION.eval_num_episodes)
        if use_server:   # episode-major: every task's first episodes start together
            units = [(t, p, i) for i in range(n_episodes) for t in pending_tasks for p in phases
                     if task_rates[t][p] is None]
        else:
            units = [(t, p, None) for t in pending_tasks for p in phases if task_rates[t][p] is None]

        def episode_file(t: str, p: str, i: int) -> Path:
            return run_output_dir / t / f"_episodes_{p}" / f"{i:03d}.json"

        claim_dir = run_output_dir / ".claims"
        shutil.rmtree(claim_dir, ignore_errors=True)   # claims of a previous (dead) manager run
        claim_dir.mkdir()
        units_file = run_output_dir / ".worker_units.json"
        units_file.write_text(json.dumps({"claim_dir": str(claim_dir), "units": [
            {"task_name": t, "task_config": phase_to_task_config[p], "eval_output_dir": str(run_output_dir / t),
             **({"episode": i, "result_file": str(episode_file(t, p, i))} if i is not None else
                {"result_file": str(run_output_dir / t / _phase_result_filename(p))})}
            for t, p, i in units]}, indent=1))
        remaining = {(t, p) for t, p, _ in units}

        def merge_episodes(t: str, p: str) -> None:
            """Write the task's _result file (as script/eval_policy.py run_task) once every episode is in."""
            files = [episode_file(t, p, i) for i in range(n_episodes)]
            if not all(f.exists() for f in files):
                return
            episodes = [json.loads(f.read_text()) for f in files]
            rate = sum(bool(e["success"]) for e in episodes) / n_episodes
            (run_output_dir / t / f"episodes_{p}.json").write_text(json.dumps(episodes, indent=1))
            (run_output_dir / t / _phase_result_filename(p)).write_text(
                f"Timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n\n"
                f"Instruction Type: {cfg.EVALUATION.instruction_type}\n\n{rate}")

        def harvest(final: bool) -> None:
            for t, p in sorted(remaining):
                result_file = run_output_dir / t / _phase_result_filename(p)
                if use_server and not result_file.exists():
                    try:
                        merge_episodes(t, p)
                    except Exception as exc:   # an episode file being replaced; next poll
                        if final:
                            log(f"episode merge failed: task={t}, phase={p}, error={repr(exc)}")
                if not result_file.exists():
                    continue
                try:
                    task_rates[t][p] = _parse_success_rate(result_file)
                except Exception as exc:
                    if final:
                        log(f"result parse failed: task={t}, phase={p}, error={repr(exc)}")
                    continue   # possibly still being written; next poll
                remaining.discard((t, p))
                log(f"done task={t} phase={p} success_rate={task_rates[t][p]:.4f}")

        servers: dict[int, tuple[Path, subprocess.Popen[str]]] = {}

        def killpg(proc: subprocess.Popen[str], sig: int) -> None:
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                pass

        def stop_servers() -> None:
            # Each server runs in its own session: the signal reaches eval_robotwin_single.py and
            # the serve_robotwin_policy.py process under it, so the GPU memory is freed now.
            for _, proc in servers.values():
                killpg(proc, signal.SIGTERM)
            deadline = time.time() + TERMINATE_TIMEOUT_SEC
            for addr, proc in servers.values():
                try:
                    proc.wait(timeout=max(0.0, deadline - time.time()))
                except subprocess.TimeoutExpired:
                    pass
                killpg(proc, signal.SIGKILL)   # the server itself, if it outlived its parent
                proc.wait()
                addr.unlink(missing_ok=True)
            servers.clear()

        atexit.register(stop_servers)   # also when the manager dies of an exception
        if use_server:
            envs_per_gpu = int(cfg.MULTIRUN.get("envs_per_gpu", 4))
            os.environ.setdefault(AUTHKEY_ENV, os.urandom(16).hex())   # inherited by servers and workers
            for gpu_id in gpu_ids:
                addr = run_output_dir / f".policy_server_gpu{gpu_id}.json"
                addr.unlink(missing_ok=True)
                cmd = [sys.executable, str(SINGLE_ENTRY), f"ckpt={str(ckpt_path)}", f"gpu_id={gpu_id}",
                       f"EVALUATION.serve_address_file={str(addr)}", f"EVALUATION.output_dir={str(output_dir)}",
                       *extra_overrides]
                log(f"launch policy server gpu={gpu_id} cmd={' '.join(cmd)}")
                servers[gpu_id] = (addr, subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), text=True, start_new_session=True))

        # Workers start right away; with a policy server they wait for its address file.
        n_workers = min(num_gpus * (envs_per_gpu if use_server else max_tasks_per_gpu), len(units))
        for k in range(n_workers):
            gpu_id = gpu_ids[k % num_gpus]
            cmd = [sys.executable, str(SINGLE_ENTRY), f"ckpt={str(ckpt_path)}", f"gpu_id={gpu_id}",
                   f"EVALUATION.worker_units_file={str(units_file)}", f"EVALUATION.output_dir={str(output_dir)}",
                   *([f"EVALUATION.policy_server={str(servers[gpu_id][0])}"] if use_server else []),
                   *extra_overrides]
            log(f"launch worker={k} gpu={gpu_id} units={len(units)} cmd={' '.join(cmd)}")
            running_states.append(RunningState(task_name=f"worker{k}", gpu_id=gpu_id, phase="*",
                                               process=subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), text=True)))
        while len(running_states) > 0:
            harvest(final=False)
            for gpu_id, (_, proc) in servers.items():
                if proc.poll() is not None:
                    has_failure = True
                    failure_message = f"policy server on gpu={gpu_id} exited with return_code={proc.returncode}"
                    failed_records.append({"task_name": f"server_gpu{gpu_id}", "phase": "*", "gpu_id": gpu_id,
                                           "return_code": proc.returncode, "reason": "server_failed"})
                    log(failure_message)
                    terminate_all_running()
                    running_states.clear()
                    break
            for state in list(running_states):
                return_code = state.process.poll()
                if return_code is None:
                    continue
                running_states.remove(state)
                if return_code != 0:
                    has_failure = True
                    failure_message = f"worker failed: {state.task_name}, gpu={state.gpu_id}, return_code={return_code}"
                    failed_records.append({"task_name": state.task_name, "phase": "*", "gpu_id": state.gpu_id,
                                           "return_code": return_code, "reason": "process_failed"})
                    log(failure_message)
                    terminate_all_running()
                    running_states.clear()
                    break
            if not has_failure and len(running_states) > 0:
                time.sleep(POLL_INTERVAL_SEC)
        stop_servers()
        harvest(final=True)
        if not has_failure and remaining:
            has_failure = True
            failure_message = f"workers exited with unfinished units: {sorted(remaining)}"
            log(failure_message)
        pending_tasks = deque(t for t in tasks if any(task_rates[t][p] is None for p in phases))
        if not has_failure:
            pending_tasks.clear()

    # Launch initial tasks for each GPU up to capacity (one process per task; nothing is
    # pending here after persistent workers finished, and nothing launches after a failure).
    if not has_failure:
        for gpu_id in gpu_ids:
            try_launch_pending(gpu_id)

    while len(running_states) > 0:
        progressed = False
        for state in list(running_states):
            gpu_id = state.gpu_id
            return_code = state.process.poll()
            if return_code is None:
                continue
            progressed = True
            running_states.remove(state)

            if return_code != 0:
                has_failure = True
                failure_message = (
                    f"worker failed: task={state.task_name}, phase={state.phase}, "
                    f"gpu={gpu_id}, return_code={return_code}"
                )
                failed_records.append(
                    {
                        "task_name": state.task_name,
                        "phase": state.phase,
                        "gpu_id": gpu_id,
                        "return_code": return_code,
                        "reason": "process_failed",
                    }
                )
                log(failure_message)
                terminate_all_running()
                running_states.clear()
                break

            result_file = run_output_dir / state.task_name / _phase_result_filename(state.phase)
            try:
                success_rate = _parse_success_rate(result_file)
            except Exception as exc:
                has_failure = True
                failure_message = (
                    f"result parse failed: task={state.task_name}, phase={state.phase}, "
                    f"gpu={gpu_id}, error={repr(exc)}"
                )
                failed_records.append(
                    {
                        "task_name": state.task_name,
                        "phase": state.phase,
                        "gpu_id": gpu_id,
                        "return_code": return_code,
                        "reason": "result_parse_failed",
                    }
                )
                log(failure_message)
                terminate_all_running()
                running_states.clear()
                break

            task_rates[state.task_name][state.phase] = success_rate
            log(
                f"done task={state.task_name} phase={state.phase} gpu={gpu_id} "
                f"success_rate={success_rate:.4f}"
            )

            next_phase = _next_phase(state.phase)
            # Skip phases that already have results loaded from disk (resume).
            while next_phase is not None and task_rates[state.task_name][next_phase] is not None:
                next_phase = _next_phase(next_phase)
            if next_phase is not None:
                running_states.append(launch_phase(
                    task_name=state.task_name,
                    gpu_id=gpu_id,
                    phase=next_phase,
                ))
                continue

            try_launch_pending(gpu_id)

        if has_failure:
            break
        if not progressed:
            time.sleep(POLL_INTERVAL_SEC)

    # Mark not started tasks when failure happened.
    if has_failure:
        for task_name in pending_tasks:
            failed_records.append(
                {
                    "task_name": task_name,
                    "phase": "not_started",
                    "gpu_id": -1,
                    "return_code": -1,
                    "reason": "aborted_not_started",
                }
            )

    write_outputs()
    log(f"summary saved: {summary_csv} and {summary_json}")

    if has_failure:
        raise RuntimeError(failure_message)

    log("manager finished successfully")


if __name__ == "__main__":
    main()
