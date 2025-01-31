"""
An example that shows how to use SampleFactory with a PettingZoo env.

Example command line for tictactoe_v3:
python -m sf_examples.train_pettingzoo_env --algo=APPO --use_rnn=False --num_envs_per_worker=20 --policy_workers_per_policy=2 --recurrence=1 --with_vtrace=False --batch_size=512 --save_every_sec=10 --experiment_summaries_interval=10 --experiment=example_pettingzoo_tictactoe_v3 --env=tictactoe_v3
python -m sf_examples.enjoy_pettingzoo_env --algo=APPO --experiment=example_pettingzoo_tictactoe_v3 --env=tictactoe_v3

"""

import sys
from typing import Optional
from dark_chess.envs.dark_chess import DarkChessGame
from dark_chess.envs.wrappers import ActionMask2ObservationWrapper

from pettingzoo.utils import turn_based_aec_to_parallel
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.envs.pettingzoo_envs import PettingZooParallelEnv
from sample_factory.train import run_rl


def make_pettingzoo_env(
    full_env_name, cfg=None, env_config=None, render_mode: Optional[str] = None
):
    env = DarkChessGame(render_mode=render_mode)
    env = ActionMask2ObservationWrapper(env)
    env = turn_based_aec_to_parallel(env)
    return PettingZooParallelEnv(env)


def register_custom_components():
    register_env("dark_chess", make_pettingzoo_env)


def parse_custom_args(argv=None, evaluation=False):
    parser, cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    cfg = parse_full_cfg(parser, argv)
    return cfg


def main():
    """Script entry point."""
    register_custom_components()
    cfg = parse_custom_args()
    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
