"""FlexPi RoboTwin policy server: load one checkpoint (at one NFE) and answer the replan requests
of any number of simulator workers until terminated.

Started by run_robotwin_manager.py (MULTIRUN.policy_server=true) through eval_robotwin_single.py
with EVALUATION.serve_address_file set, which passes the same arguments script/eval_policy.py gets:

    python experiments/robotwin/serve_robotwin_policy.py --address-file F \
        --config policy/flexpi_policy/deploy_policy.yml --overrides --ckpt_setting ... --num_inference_steps 4 ...

The model is built by the same ``get_model`` the in-process eval uses; see
flexpi_policy/policy_server.py for the protocol. FLEXPI_POLICY_AUTHKEY must be set.
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))   # experiments/robotwin: the flexpi_policy package

from flexpi_policy.deploy_policy import get_model  # noqa: E402
from flexpi_policy.policy_server import authkey_from_env, serve  # noqa: E402


def parse_args_and_config():
    """As script/eval_policy.py: a YAML config updated by ``--key value`` pairs (values eval'd)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--address-file", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    pairs = args.overrides or []
    for i in range(0, len(pairs), 2):
        value = pairs[i + 1]
        try:
            value = eval(value)
        except Exception:
            pass
        config[pairs[i].lstrip("--")] = value
    return args.address_file, config


def main():
    address_file, usr_args = parse_args_and_config()
    usr_args.pop("policy_server", None)   # this process is the server
    authkey = authkey_from_env()
    serve(get_model(usr_args), address_file, authkey)


if __name__ == "__main__":
    main()
