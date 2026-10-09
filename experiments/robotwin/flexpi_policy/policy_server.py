"""FlexPi RoboTwin policy server and its client.

The server holds one loaded ``WorldActionRobotWinPolicy`` (one checkpoint, one NFE) and answers
the requests of any number of simulator processes, one request at a time in arrival order. Each
request runs exactly the in-process code path (``infer_chunk`` -> ``_infer_action_chunk``, the
policy's fixed noise seed on every call), so a served rollout matches an in-process one.

``RemotePolicy`` is the simulator side: the same interface as the in-process policy for
script/eval_policy.py (``step``, ``reset``, ``should_request_observation``, ``prediction_frames``),
with the action queue kept locally and one round trip per replan.

Transport: ``multiprocessing.connection`` over TCP on 127.0.0.1 with an HMAC authkey (passed in
the environment, FLEXPI_POLICY_AUTHKEY, never on the command line). Messages carry numpy arrays
only: torch tensors would be pickled as shared-memory handles.
"""
import json
import logging
import os
import queue
import threading
import time
import traceback
from collections import deque
from multiprocessing.connection import Client, Listener
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

AUTHKEY_ENV = "FLEXPI_POLICY_AUTHKEY"
_OBS_CAMERAS = ("head_camera", "left_camera", "right_camera")   # the model's three views
_OBS_FIELDS = ("rgb", "depth", "intrinsic_cv")


def authkey_from_env() -> bytes:
    key = os.environ.get(AUTHKEY_ENV)
    if not key:
        raise RuntimeError(f"{AUTHKEY_ENV} is not set (run_robotwin_manager.py sets it for servers and workers)")
    return key.encode()


def slim_observation(observation):
    """The part of a RoboTwin observation the policy reads (three cameras, joint state)."""
    obs = observation["observation"]
    return {"observation": {cam: {f: obs[cam][f] for f in _OBS_FIELDS} for cam in _OBS_CAMERAS},
            "joint_action": {"vector": np.asarray(observation["joint_action"]["vector"])}}


def frames_per_step(starts, clips, num_steps, action_horizon):
    """Per executed step, the decoded predicted frame that shows it (or None).

    The replan at step ``starts[r]`` predicted ``clips[r]``, whose frame j shows step
    ``starts[r] + j * stride`` with stride = action_horizon / (frames - 1) (frame 0 is the current
    observation); it covers the steps up to the next replan.
    """
    out = [None] * num_steps
    for r, (k0, clip) in enumerate(zip(starts, clips)):
        k1 = starts[r + 1] if r + 1 < len(starts) else num_steps
        stride = action_horizon / max(len(clip) - 1, 1)
        for k in range(k0, min(k1, num_steps)):
            out[k] = clip[min(int(round((k - k0) / stride)), len(clip) - 1)]
    return out


# ---------------------------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------------------------

def _dispatch(policy, request):
    op = request.get("op")
    if op == "infer":
        chunk, latents = policy.infer_chunk(request["obs"], request["instruction"],
                                            return_latents=bool(request.get("latents")))
        return {"chunk": np.asarray(chunk),
                "latents": None if latents is None else latents.float().cpu().numpy()}
    if op == "decode":
        import torch
        latents = [torch.from_numpy(a).to(torch.bfloat16) for a in request["latents"]]   # bf16 -> f32 -> bf16 is exact
        return {"clips": policy.decode_head_clips(latents, tuple(request["frame_hw"]))}
    if op == "info":
        return {"action_horizon": policy.action_horizon, "replan_steps": policy.replan_steps,
                "num_inference_steps": policy.num_inference_steps, "seed": policy.seed}
    raise ValueError(f"unknown request op {op!r}")


def serve(policy, address_file, authkey: bytes, stats_every_s: float = 60.0) -> None:
    """Answer requests forever (the manager terminates the process). The listening address is
    written to ``address_file`` once the policy is loaded, which is the readiness signal."""
    import torch

    listener = Listener(("127.0.0.1", 0), authkey=authkey)
    requests: "queue.Queue" = queue.Queue()

    def handle(conn):
        # Clients are synchronous (one request in flight), so this thread both reads the
        # connection and sends the reply the main thread computed.
        try:
            while True:
                request = conn.recv()
                box = {"done": threading.Event(), "t_in": time.perf_counter()}
                requests.put((request, box))
                box["done"].wait()
                conn.send(box["reply"])
        except (EOFError, OSError):
            pass
        finally:
            conn.close()

    def accept():
        while True:
            try:
                conn = listener.accept()
            except Exception as exc:   # failed handshake (wrong key) or a dropped connect
                print(f"[policy server] rejected a connection: {exc!r}", flush=True)
                continue
            threading.Thread(target=handle, args=(conn,), daemon=True).start()

    threading.Thread(target=accept, daemon=True).start()
    host, port = listener.address
    address_file = Path(address_file)
    tmp = address_file.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps({"host": host, "port": port, "pid": os.getpid()}))
    os.replace(tmp, address_file)
    print(f"[policy server] ready on {host}:{port} (pid {os.getpid()}); address in {address_file}", flush=True)

    n = busy = wait = 0.0
    window_t0 = time.perf_counter()
    while True:
        try:
            request, box = requests.get(timeout=stats_every_s)
        except queue.Empty:
            request = None
        if request is not None:
            t0 = time.perf_counter()
            try:
                box["reply"] = {"ok": True, **_dispatch(policy, request)}
            except Exception:
                box["reply"] = {"ok": False, "error": traceback.format_exc()}
                print(f"[policy server] request {request.get('op')!r} failed:\n{box['reply']['error']}", flush=True)
            t1 = time.perf_counter()
            box["done"].set()
            n += 1
            busy += t1 - t0
            wait += t0 - box["t_in"]
        elapsed = time.perf_counter() - window_t0
        if elapsed >= stats_every_s:
            mem = (f" | GPU mem peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB allocated, "
                   f"{torch.cuda.max_memory_reserved() / 2**30:.1f} GiB reserved") if torch.cuda.is_available() else ""
            print(f"[policy server] {int(n)} requests in {elapsed:.0f} s | busy {100 * busy / elapsed:.0f}% | "
                  f"mean service {1000 * busy / max(n, 1):.0f} ms, queue wait {1000 * wait / max(n, 1):.0f} ms | "
                  f"queued {requests.qsize()}{mem}", flush=True)
            n = busy = wait = 0.0
            window_t0 = time.perf_counter()


# ---------------------------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------------------------

class RemotePolicy:
    """Simulator-side policy that sends each replan to a policy server (see module docstring)."""

    def __init__(self, address_file, authkey: bytes, timing_enabled: bool = False, wait_s: float = 3600.0):
        address_file = Path(address_file)
        deadline = time.time() + wait_s
        while not address_file.exists():   # the server writes it once its model is loaded
            if time.time() > deadline:
                raise TimeoutError(f"no policy server address in {address_file} after {wait_s:.0f} s")
            time.sleep(2)
        address = json.loads(address_file.read_text())
        self._conn = Client((address["host"], address["port"]), authkey=authkey)
        info = self._call({"op": "info"})
        self.action_horizon = int(info["action_horizon"])
        self.replan_steps = int(info["replan_steps"])
        self.num_inference_steps = int(info["num_inference_steps"])
        self.seed = info["seed"]
        print(f"[FlexPi] remote policy at {address['host']}:{address['port']} | horizon={self.action_horizon} "
              f"replan={self.replan_steps} NFE={self.num_inference_steps}", flush=True)
        self.timing_enabled = bool(timing_enabled)
        self.pending_actions: deque = deque()
        self.record_predictions = False   # set per episode by script/eval_policy.py
        self._predictions = []            # (step, video latents) per replan of a recorded episode
        self.keep_last_pred = False
        self._last_chunk = None
        self.episode_count = 0
        self.step_count = 0
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}
        self._infer_calls = 0
        self._episode_start_time = time.perf_counter()

    def _call(self, request):
        self._conn.send(request)
        reply = self._conn.recv()
        if not reply.pop("ok"):
            raise RuntimeError(f"policy server error on {request.get('op')!r}:\n{reply['error']}")
        return reply

    def should_request_observation(self) -> bool:
        return not self.pending_actions

    def step(self, task_env, observation) -> None:
        if not self.pending_actions:
            if observation is None:
                raise ValueError("Observation is required when action queue is empty (replan step for flexpi).")
            t0 = time.perf_counter()
            reply = self._call({"op": "infer", "obs": slim_observation(observation),
                                "instruction": task_env.get_instruction(), "latents": self.record_predictions})
            if self.timing_enabled:   # round trip: queue wait + inference + transfer
                self._timing_rollout["infer_s"] += time.perf_counter() - t0
                self._infer_calls += 1
            chunk = reply["chunk"]
            self._last_chunk = chunk
            if self.record_predictions and reply["latents"] is not None:
                self._predictions.append((self.step_count, reply["latents"]))
            for i in range(min(self.replan_steps, chunk.shape[0])):
                self.pending_actions.append(np.asarray(chunk[i], dtype=np.float32))
        if not self.pending_actions:
            logger.warning("No action generated; skip current eval step.")
            return
        action = self.pending_actions.popleft()
        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        task_env.take_action(action, action_type="qpos")
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0
        self.step_count += 1

    def prediction_frames(self, num_steps, frame_hw):
        """Per executed step, the decoded prediction of the head camera (decoded on the server)."""
        if not self._predictions:
            return None
        clips = self._call({"op": "decode", "latents": [lat for _, lat in self._predictions],
                            "frame_hw": tuple(frame_hw)})["clips"]
        return frames_per_step([k0 for k0, _ in self._predictions], clips, num_steps, self.action_horizon)

    def render_divergence(self, *args, **kwargs):
        raise NotImplementedError("rendering diagnostics (RENDER_SHADOW_SPP) need the in-process policy, "
                                  "not a policy server")

    def reset(self) -> None:
        if self.timing_enabled and self.step_count > 0:
            wall_s = time.perf_counter() - self._episode_start_time
            n_calls = max(self._infer_calls, 1)
            infer_ms_per_call = 1000.0 * self._timing_rollout["infer_s"] / n_calls
            sim_ms_per_step = 1000.0 * self._timing_rollout["sim_s"] / max(self.step_count, 1)
            policy_ms_per_step = infer_ms_per_call / self.replan_steps + sim_ms_per_step
            print(f"[FlexPi timing] episode={self.episode_count} steps={self.step_count} "
                  f"wall_s={wall_s:.3f} fps={self.step_count / wall_s:.2f} "
                  f"policy_fps={1000.0 / policy_ms_per_step if policy_ms_per_step > 0 else float('inf'):.2f} "
                  f"infer_calls={self._infer_calls} infer_s={self._timing_rollout['infer_s']:.3f} "
                  f"infer_ms/call={infer_ms_per_call:.1f} sim_s={self._timing_rollout['sim_s']:.3f}", flush=True)
        self.pending_actions.clear()
        self._predictions = []
        self.episode_count += 1
        self.step_count = 0
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}
        self._infer_calls = 0
        self._episode_start_time = time.perf_counter()
