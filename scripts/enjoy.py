import sys
from typing import Optional
from dark_chess.envs.dark_chess import DarkChessGame
from dark_chess.envs.wrappers import ActionMask2ObservationWrapper, RewardShapingWrapper

from pettingzoo.utils import turn_based_aec_to_parallel
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.envs.pettingzoo_envs import PettingZooParallelEnv
from sample_factory.enjoy import enjoy


def make_pettingzoo_env(
    full_env_name, cfg=None, env_config=None, render_mode: Optional[str] = None
):
    env = DarkChessGame(render_mode=render_mode)
    env = RewardShapingWrapper(env)
    env = ActionMask2ObservationWrapper(env)
    env = turn_based_aec_to_parallel(env)
    return PettingZooParallelEnv(env)


def register_custom_components():
    register_env("dark_chess", make_pettingzoo_env)


def parse_custom_args(argv=None, evaluation=False):
    parser, cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    cfg = parse_full_cfg(parser, argv)
    return cfg


def main():  # pragma: no cover
    """Script entry point."""
    register_custom_components()
    cfg = parse_custom_args(evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
