import sys
import os
import subprocess

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import json
import random
import time
import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


EVAL_VIDEO_KEEP = int(os.environ.get("EVAL_VIDEO_KEEP", "2"))      # per outcome, per task run
EVAL_VIDEO_FPS = int(os.environ.get("EVAL_VIDEO_FPS", "25"))       # executed actions per second of video
EVAL_VIDEO_STRIDE = max(1, int(os.environ.get("EVAL_VIDEO_STRIDE", "4")))  # one frame every N actions (as envs/_base_task.py)
# Rendering diagnostics (off by default). RENDER_LOG_DIR: one JSONL per task with every replan's
# state, executed chunk and render time, and every episode's outcome. RENDER_SHADOW_SPP: also
# re-render each replan's state at this ray-tracing spp (shadow cameras, envs/_base_task.py) and log how the policy's outputs diverge
# (model.render_divergence); RENDER_SHADOW_DECODE=0 skips decoding the predicted videos.
# Static cameras each observation renders: the policy reads the head view only (plus the wrist
# cameras), so the embodiment's front camera is not ray-traced. "all" renders every static camera.
EVAL_STATIC_CAMERAS = os.environ.get("EVAL_STATIC_CAMERAS", "head_camera")
RENDER_LOG_DIR = os.environ.get("RENDER_LOG_DIR") or None
RENDER_SHADOW_SPP = int(os.environ.get("RENDER_SHADOW_SPP", "0") or 0)
RENDER_SHADOW_DECODE = os.environ.get("RENDER_SHADOW_DECODE", "1") != "0"


def write_eval_video(path, frames, predicted=None, nfe=None, fps=None):
    """Head-camera rollout, optionally beside the model's decoded prediction of the same view."""
    import cv2
    h, w = frames[0].shape[:2]
    gap = np.full((h, 4, 3), 255, dtype=np.uint8)
    out = []
    for i, frame in enumerate(frames):
        tile = frame
        if predicted is not None:
            pred = predicted[i] if i < len(predicted) and predicted[i] is not None else np.zeros_like(frame)
            tile = np.concatenate([frame, gap, pred], axis=1)
        out.append(np.ascontiguousarray(tile))
    if predicted is not None:
        label = "model prediction" + (f" (NFE {nfe})" if nfe else "")
        for tile in out:   # white text on a dark outline: readable on the bright scenes
            for text, x in (("simulation", 6), (label, w + 10)):
                cv2.putText(tile, text, (x, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(tile, text, (x, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    oh, ow = out[0].shape[:2]
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                             "-video_size", f"{ow}x{oh}", "-framerate", f"{fps or EVAL_VIDEO_FPS:g}", "-i", "-",
                             "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "23", str(path)],
                            stdin=subprocess.PIPE)
    for tile in out:
        proc.stdin.write(tile.tobytes())
    proc.stdin.close()
    proc.wait()


def get_eval_video_size(args):
    head_camera_cfg = get_camera_config(args["camera"]["head_camera_type"])
    video_w = int(head_camera_cfg["w"])
    video_h = int(head_camera_cfg["h"])

    if args["camera"].get("collect_wrist_camera", False):
        wrist_camera_cfg = get_camera_config(args["camera"]["wrist_camera_type"])
        wrist_w = int(wrist_camera_cfg["w"])
        wrist_h = int(wrist_camera_cfg["h"])
        video_w = max(video_w, wrist_w * 2)
        video_h = video_h + wrist_h

    return f"{video_w}x{video_h}"


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def _result_suffix_from_task_config(task_config):
    if task_config == "demo_clean":
        return "clean"
    if task_config == "demo_randomized":
        return "random"
    raise ValueError(
        f"Unsupported `task_config` for fixed result naming: {task_config}. "
        "Expected one of: ['demo_clean', 'demo_randomized']."
    )


class ExpertCache:
    """Outcome of RoboTwin's expert check per seed, on disk and shared by every evaluated model.

    The check (scripted planner run before each policy episode) decides which seeds are
    episodes, and its episode info decides the instruction; it does not depend on the policy.
    Caching it makes every model see the same seeds (cuRobo planning is not deterministic
    run to run) and skips the planner run and one scene setup per episode. The first outcome
    written for a seed is final, also when the check raised an error (recorded in "error"):
    workers that evaluate episodes of one task in parallel must all agree on which seeds are
    episodes. ``claim`` / ``release`` keep two workers from running the same check at once."""

    LOCK_STALE_S = 1800   # a compute lock older than this is abandoned (its worker died)

    def __init__(self, root, task_name, task_config, embodiment):
        self.dir = Path(root) / task_name / task_config / embodiment if root else None

    def get(self, seed):
        if self.dir is None:
            return None
        try:
            return json.loads((self.dir / f"{seed}.json").read_text())
        except (FileNotFoundError, ValueError):
            return None

    def put(self, seed, ok, info=None, error=None):
        """Store the outcome unless one exists; returns the stored entry (None if not storable)."""
        if self.dir is None:
            return None
        try:
            text = json.dumps({"ok": bool(ok), "info": info, **({"error": error} if error else {})})
        except TypeError:
            return None   # info not JSON-serializable: leave the seed uncached
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / f".{seed}.json.tmp{os.getpid()}"
        tmp.write_text(text)
        try:
            os.link(tmp, self.dir / f"{seed}.json")   # atomic, fails if another worker stored it first
        except FileExistsError:
            pass
        finally:
            tmp.unlink()
        return self.get(seed)

    def claim(self, seed):
        """True if this process may run the check of ``seed`` (False: another one is running it)."""
        if self.dir is None:
            return True
        self.dir.mkdir(parents=True, exist_ok=True)
        lock = self.dir / f".{seed}.lock"
        try:
            lock.mkdir()
            return True
        except FileExistsError:
            try:
                stale = time.time() - lock.stat().st_mtime > self.LOCK_STALE_S
            except FileNotFoundError:
                return self.claim(seed)   # released meanwhile
            if stale:
                print(f"[expert cache] taking over the stale check of seed {seed}")
                lock.touch()
                return True
            return False

    def release(self, seed):
        if self.dir is not None:
            try:
                (self.dir / f".{seed}.lock").rmdir()
            except FileNotFoundError:
                pass


def expert_check(TASK_ENV, args, now_id, seed, expert_cache):
    """(ok, episode_info) of RoboTwin's expert check for ``seed``, through the cache; None when
    another worker is running this seed's check (the cache will hold it)."""
    cached = expert_cache.get(seed) if expert_cache is not None else None
    if cached is not None:
        return cached["ok"], {"info": cached["info"]}
    if expert_cache is not None and not expert_cache.claim(seed):
        return None
    render_freq = args["render_freq"]
    args["render_freq"] = 0
    ok, episode_info, error = False, None, None
    try:
        TASK_ENV.setup_demo(now_ep_num=now_id, seed=seed, is_test=True, **args)
        episode_info = TASK_ENV.play_once()
        TASK_ENV.close_env()
        ok = bool(TASK_ENV.plan_success and TASK_ENV.check_success())
    except UnStableError as e:
        TASK_ENV.close_env()
        error = "UnStableError"
    except Exception as e:
        print(" -------------")
        print("Error: ", e)
        print("Stack Trace: ", traceback.format_exc())
        print(" -------------")
        TASK_ENV.close_env()
        print("error occurs !")
        error = repr(e)
    finally:
        args["render_freq"] = render_freq
    if expert_cache is not None:
        entry = expert_cache.put(seed, ok, episode_info["info"] if ok else None, error)
        expert_cache.release(seed)
        if entry is not None:   # the stored outcome (another worker's, if it stored first)
            return entry["ok"], {"info": entry["info"]}
    return ok, episode_info


def run_episode(TASK_ENV, args, model, task_name, now_id, now_seed, episode_info, instruction_type, test_num,
                skip_get_obs_within_replan, want_video, keep_video, clear_cache):
    """Roll out the policy on ``now_seed`` (whose expert check passed). Returns the success flag, or
    None when the rollout scene could not be set up. ``want_video()``: whether to capture the eval
    video (asked once the scene is set up); ``keep_video(succ)``: whether to write it (takes a slot)."""
    eval_func = eval_function_decorator(args["policy_name"], "eval")
    reset_func = eval_function_decorator(args["policy_name"], "reset_model")
    try:
        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
    except UnStableError as e:
        TASK_ENV.close_env()
        return None
    except Exception as e:
        print(" -------------")
        print("Error: ", e)
        print("Stack Trace: ", traceback.format_exc())
        print(" -------------")
        TASK_ENV.close_env()
        print("error occurs !")
        return None
    episode_info_list = [episode_info["info"]]
    # Seeded by the episode seed, so every model gets the same instruction for the same
    # episode (RoboTwin's generator shuffles with the unseeded `random` module).
    random.seed(now_seed)
    results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
    instruction = np.random.RandomState(now_seed).choice(results[0][instruction_type])
    TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

    # Eval video: frames are captured only while a video slot (EVAL_VIDEO_KEEP per outcome)
    # is still open, and then every EVAL_VIDEO_STRIDE-th action (ray-traced head camera).
    record = TASK_ENV.eval_video_path is not None and EVAL_VIDEO_KEEP > 0 and want_video()
    TASK_ENV.eval_video_record = record
    if hasattr(model, "record_predictions"):
        model.record_predictions = record   # keep the predicted video latents of every replan
    TASK_ENV.eval_video_frames = []
    succ = False
    reset_func(model)
    render_log = None
    if RENDER_LOG_DIR is not None:
        os.makedirs(RENDER_LOG_DIR, exist_ok=True)
        render_log = open(os.path.join(RENDER_LOG_DIR, f"{task_name}.jsonl"), "a")
        main_spp = int(os.environ.get("ROBOTWIN_RT_SPP", "32"))
        model.keep_last_pred = bool(RENDER_SHADOW_SPP)
        episode_t0 = time.perf_counter()
    while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
        need_obs = True
        if skip_get_obs_within_replan and hasattr(model, "should_request_observation"):
            need_obs = bool(model.should_request_observation())

        observation = shadow_observation = None
        if need_obs:
            obs_t0 = time.perf_counter() if render_log else 0.0
            observation = TASK_ENV.get_obs()
            if render_log:
                obs_ms = 1000 * (time.perf_counter() - obs_t0)
                if RENDER_SHADOW_SPP:   # same state through the shadow cameras (other spp)
                    shadow_observation = TASK_ENV.get_shadow_obs()
        replan_step = TASK_ENV.take_action_cnt
        eval_func(TASK_ENV, model, observation)
        if render_log and observation is not None:
            chunk = model._last_chunk
            rec = {"type": "replan", "task": task_name, "seed": now_seed, "episode": TASK_ENV.test_num,
                   "step": replan_step, "spp": main_spp, "get_obs_ms": round(obs_ms, 2),
                   "state": np.round(np.asarray(observation["joint_action"]["vector"], np.float64), 6).tolist(),
                   "chunk_exec": np.round(chunk[:model.replan_steps], 6).tolist()}
            if shadow_observation is not None:
                rec["shadow_spp"] = RENDER_SHADOW_SPP
                rec.update(model.render_divergence(observation, shadow_observation, TASK_ENV.get_instruction(),
                                                   decode=RENDER_SHADOW_DECODE))
            render_log.write(json.dumps(rec) + "\n")
            render_log.flush()
        if TASK_ENV.eval_success:
            succ = True
            break
    if render_log:
        render_log.write(json.dumps({"type": "episode", "task": task_name, "seed": now_seed,
                                     "episode": TASK_ENV.test_num, "spp": main_spp, "success": bool(succ),
                                     "steps": int(TASK_ENV.take_action_cnt), "instruction": TASK_ENV.get_instruction(),
                                     "wall_s": round(time.perf_counter() - episode_t0, 2)}) + "\n")
        render_log.close()
    # task_total_reward += TASK_ENV.episode_score
    if TASK_ENV.eval_video_path is not None:
        if TASK_ENV.eval_video_frames and keep_video(succ):
            is_randomized = "randomized" in str(args["task_config"]).lower()
            video_path = (Path(TASK_ENV.eval_video_path) /
                          f"episode{TASK_ENV.test_num}_randomized-{str(is_randomized).lower()}_success-{str(succ).lower()}.mp4")
            steps = [k for k, _ in TASK_ENV.eval_video_frames]
            frames = [f for _, f in TASK_ENV.eval_video_frames]
            try:   # a video problem must never fail the evaluation
                predicted = None
                if hasattr(model, "prediction_frames"):
                    try:
                        per_step = model.prediction_frames(steps[-1] + 1, frames[0].shape[:2])
                        predicted = None if per_step is None else [per_step[k] for k in steps]
                    except Exception:
                        print("[eval video] prediction decode failed; writing the rollout alone\n"
                              + traceback.format_exc())
                nfe = getattr(model, "num_inference_steps", None) or args.get("num_inference_steps")
                write_eval_video(video_path, frames, predicted, nfe, fps=EVAL_VIDEO_FPS / EVAL_VIDEO_STRIDE)
            except Exception:
                print("[eval video] writing failed\n" + traceback.format_exc())
        TASK_ENV.eval_video_frames = []

    if succ:
        print("\033[92mSuccess!\033[0m")
    else:
        print("\033[91mFail!\033[0m")
    TASK_ENV.close_env(clear_cache=clear_cache)
    if TASK_ENV.render_freq:
        TASK_ENV.viewer.close()
    return succ


def build_task_args(usr_args, task_name, task_config):
    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = usr_args["ckpt_setting"]
    args["obs_static_cameras"] = None if EVAL_STATIC_CAMERAS == "all" else EVAL_STATIC_CAMERAS.split(",")

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])
    return args, embodiment_name


def run_task(usr_args, model, task_name, task_config, eval_output_dir):
    """Evaluate one (task, task_config) with an already-loaded policy; writes its _result file."""
    eval_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_setting = usr_args["ckpt_setting"]
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    skip_get_obs_within_replan = parse_bool(usr_args.get("skip_get_obs_within_replan", False))
    eval_num_episodes = int(usr_args.get("eval_num_episodes", 100))
    if eval_num_episodes <= 0:
        raise ValueError(f"`eval_num_episodes` must be > 0, got: {eval_num_episodes}")
    save_dir = None
    video_save_dir = None
    video_size = None

    args, embodiment_name = build_task_args(usr_args, task_name, task_config)

    if eval_output_dir is not None and str(eval_output_dir).strip() != "":
        save_dir = Path(str(eval_output_dir))
    else:
        save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{eval_ts}")
    save_dir.mkdir(parents=True, exist_ok=True)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        video_size = get_eval_video_size(args)
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name

    seed = usr_args["seed"]

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = eval_num_episodes
    topk = 1

    expert_cache = ExpertCache(usr_args.get("expert_cache_dir"), task_name, task_config, embodiment_name)
    st_seed, suc_num = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type,
                                   skip_get_obs_within_replan=skip_get_obs_within_replan,
                                   expert_cache=expert_cache)
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    result_suffix = _result_suffix_from_task_config(task_config)
    file_path = os.path.join(save_dir, f"_result_{result_suffix}.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {eval_ts}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")
    # return task_reward


def episode_seed(TASK_ENV, args, expert_cache, st_seed, index, lookahead=8):
    """(seed, episode_info) of what the per-task loop runs as episode ``index``: the index-th seed
    from ``st_seed`` whose expert check passes. Missing checks are run here; while another worker
    runs the check of a seed, up to ``lookahead`` later unchecked seeds are checked meanwhile, so
    the workers on one task plan its seeds in parallel."""
    passed, seed = -1, st_seed
    while True:
        entry = expert_cache.get(seed)
        if entry is None:
            checked = expert_check(TASK_ENV, args, passed + 1, seed, expert_cache)
            if checked is None:   # another worker has this seed: check a later one meanwhile, or wait
                for ahead in range(seed + 1, seed + 1 + lookahead):
                    if (expert_cache.get(ahead) is None
                            and expert_check(TASK_ENV, args, passed + 1, ahead, expert_cache) is not None):
                        break
                else:
                    print(f"[expert cache] waiting for the check of seed {seed} (another worker)", flush=True)
                    time.sleep(10)
                continue
            entry = {"ok": checked[0], "info": checked[1]["info"] if checked[1] else None}
        if entry["ok"]:
            passed += 1
            if passed == index:
                return seed, {"info": entry["info"]}
        seed += 1


_ROLLOUTS = [0]   # rollouts run by this process (renderer cache cleared every clear_cache_freq)


def run_episode_unit(usr_args, model, unit, task_envs):
    """Episode ``unit['episode']`` of a task, exactly as the per-task loop runs it (same seed,
    instruction and rollout), written to ``unit['result_file']`` (JSON). Workers evaluate the
    episodes of a task in parallel; run_robotwin_manager.py merges them into the task's _result
    file. Eval videos: the first EVAL_VIDEO_KEEP successes and failures to finish (video slots)."""
    task_name, task_config, index = unit["task_name"], unit["task_config"], int(unit["episode"])
    eval_num_episodes = int(usr_args.get("eval_num_episodes", 100))
    args, embodiment_name = build_task_args(usr_args, task_name, task_config)
    save_dir = Path(unit["eval_output_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    if args["eval_video_log"]:
        args["eval_video_save_dir"] = save_dir
    args["policy_name"] = usr_args["policy_name"]
    args["eval_mode"] = True
    if task_name not in task_envs:
        task_envs[task_name] = class_decorator(task_name)
    TASK_ENV = task_envs[task_name]

    expert_cache = ExpertCache(usr_args.get("expert_cache_dir"), task_name, task_config, embodiment_name)
    if expert_cache.dir is None:
        raise ValueError("episode units need the shared expert cache (EVALUATION.expert_cache_dir)")
    seed, episode_info = episode_seed(TASK_ENV, args, expert_cache, 100000 * (1 + usr_args["seed"]), index)
    TASK_ENV.test_num = index   # names the eval video and the render log's episode, as in the per-task loop

    slots = save_dir / ".video_slots"
    def keep_video(succ):
        slots.mkdir(exist_ok=True)
        for k in range(EVAL_VIDEO_KEEP):
            try:
                (slots / f"{succ}_{k}").mkdir()
                return True
            except FileExistsError:
                continue
        return False
    def want_video():   # a slot of either outcome still open
        return any(not (slots / f"{o}_{k}").exists() for o in (True, False) for k in range(EVAL_VIDEO_KEEP))
    _ROLLOUTS[0] += 1
    print(f"\033[34m{task_name} episode {index}: seed {seed}\033[0m")
    succ = run_episode(TASK_ENV, args, model, task_name, index, seed, episode_info, usr_args["instruction_type"],
                       eval_num_episodes, parse_bool(usr_args.get("skip_get_obs_within_replan", False)), want_video,
                       keep_video, clear_cache=(_ROLLOUTS[0] % args["clear_cache_freq"] == 0))
    if succ is None:
        # The per-task loop would move on to the next seed and shift every later episode; the
        # units of the other episodes cannot follow that, so stop instead (rerun retries).
        raise RuntimeError(f"{task_name} episode {index}: seed {seed} passed its expert check but its "
                           "rollout scene could not be set up")
    result = {"episode": index, "seed": seed, "success": bool(succ), "steps": int(TASK_ENV.take_action_cnt),
              "instruction": TASK_ENV.get_instruction()}
    result_file = Path(unit["result_file"])
    result_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = result_file.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(result))
    os.replace(tmp, result_file)
    print(f"\033[93m{task_name}\033[0m episode {index} (seed {seed}): "
          + ("\033[92msuccess\033[0m" if succ else "\033[91mfailure\033[0m") + f" after {result['steps']} steps")


UNIT_BEGIN, UNIT_END = "@@EVAL_UNIT_BEGIN", "@@EVAL_UNIT_END"   # log markers (eval_robotwin_single splits on them)


def main(usr_args):
    policy_name = usr_args["policy_name"]
    get_model = eval_function_decorator(policy_name, "get_model")

    # One process, one model load (or one policy-server connection): either the single
    # (task_name, task_config) of the arguments, or worker mode (worker_units_file), where several
    # workers share a list of units and each claims the next unfinished one (mkdir in claim_dir is
    # atomic). A unit is a whole task, or one episode of it ("episode": run_episode_unit).
    units_file = usr_args.get("worker_units_file")
    if units_file:
        spec = json.loads(Path(units_file).read_text())
        units, claim_dir = spec["units"], Path(spec["claim_dir"])
    else:
        units = [{"task_name": usr_args["task_name"], "task_config": usr_args["task_config"],
                  "eval_output_dir": usr_args.get("eval_output_dir"), "result_file": None}]
        claim_dir = None

    args0, _ = build_task_args(usr_args, units[0]["task_name"], units[0]["task_config"])
    usr_args["left_arm_dim"] = len(args0["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args0["right_embodiment_config"]["arm_joints_name"][1])
    model = get_model(usr_args)

    task_envs = {}
    for unit in units:
        name = f"{unit['task_name']} {unit['task_config']}" + (f" {unit['episode']}" if "episode" in unit else "")
        if claim_dir is not None:
            if Path(unit["result_file"]).exists():
                continue
            try:
                os.mkdir(claim_dir / name.replace(" ", "__"))
            except FileExistsError:
                continue   # another worker has it
        print(f"{UNIT_BEGIN} {name}", flush=True)
        if "episode" in unit:
            run_episode_unit(usr_args, model, unit, task_envs)
        else:
            run_task(usr_args, model, unit["task_name"], unit["task_config"], unit["eval_output_dir"])
        print(f"{UNIT_END} {name}", flush=True)


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None,
                skip_get_obs_within_replan=False,
                expert_cache=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check_on = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True
    kept_videos = {True: 0, False: 0}

    def keep_video(succ):   # the first EVAL_VIDEO_KEEP successes and failures of this task
        if kept_videos[succ] >= EVAL_VIDEO_KEEP:
            return False
        kept_videos[succ] += 1
        return True

    while succ_seed < test_num:
        expert_ok, episode_info = True, None
        if expert_check_on:
            checked = None
            while checked is None:   # None: another worker is running this seed's check
                checked = expert_check(TASK_ENV, args, now_id, now_seed, expert_cache)
                if checked is None:
                    print(f"[expert cache] waiting for the check of seed {now_seed} (another worker)")
                    time.sleep(10)
            expert_ok, episode_info = checked

        if (not expert_check_on) or expert_ok:
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            continue

        succ = run_episode(TASK_ENV, args, model, task_name, now_id, now_seed, episode_info, instruction_type,
                           test_num, skip_get_obs_within_replan,
                           lambda: min(kept_videos.values()) < EVAL_VIDEO_KEEP, keep_video,
                           clear_cache=((succ_seed + 1) % clear_cache_freq == 0))
        if succ is None:
            # This seed passed expert_check but failed during rollout env init.
            # Roll back the accepted-seed counter and skip to next seed.
            succ_seed -= 1
            if len(suc_test_seed_list) > 0 and suc_test_seed_list[-1] == now_seed:
                suc_test_seed_list.pop()
            now_seed += 1
            continue

        if succ:
            TASK_ENV.suc += 1
        now_id += 1
        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        now_seed += 1

    return now_seed, TASK_ENV.suc


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
