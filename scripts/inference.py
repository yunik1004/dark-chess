import argparse
from pathlib import Path

# from lightning.fabric import Fabric
from dark_chess.envs.dark_chess import DarkChessGame
from dark_chess.envs.wrappers import VideoRenderingWrapper


def make_env(args: argparse.Namespace, **kwargs):
    env = DarkChessGame(cheat_mode=(False, False))
    env = VideoRenderingWrapper(
        env=env, agent_id=args.player, video_path=args.output_path, fps=args.output_fps
    )
    return env


def main(args: argparse.Namespace) -> None:
    """Main function

    Parameters
    ----------
    args : argparse.Namespace
        Arguments
    """
    # fabric = Fabric(accelerator=args.accelerator, devices="auto", strategy="auto")

    env = make_env(args)
    env.reset(seed=args.seed_env)

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            action = None
        else:
            action = env.action_space(agent).sample(
                info["action_mask"]
            )  # this is where you would insert your policy
        env.step(action)

    env.close()


def get_args() -> argparse.Namespace:
    """Parse command-line arguments

    Returns
    -------
    argparse.Namespace
        Arguments
    """
    parser = argparse.ArgumentParser("Inference script")
    parser.add_argument(
        "--accelerator", type=str, default="auto", help="inference accelerator"
    )
    parser.add_argument(
        "--output_path", type=str, default="./outputs/out.mp4", help="output video path"
    )
    parser.add_argument("--output_fps", type=int, default=2, help="output video FPS")
    parser.add_argument("--player", type=int, default=-1, help="player to focus")
    parser.add_argument("--seed_env", type=int, default=0, help="seed for environment")
    return parser.parse_args()


if __name__ == "__main__":
    opts = get_args()
    output_path = Path(opts.output_path)
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)
    main(opts)
