import argparse
import imageio

# from lightning.fabric import Fabric
from dark_chess.envs.dark_chess import DarkChessEnv


def make_env(args: argparse.Namespace, **kwargs):
    env = DarkChessEnv()
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

    recorded_frames = []

    env.reset()

    image = env.render(args.player)
    recorded_frames.append(image)

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            action = None
        else:
            action = env.action_space(agent).sample(
                info["action_mask"]
            )  # this is where you would insert your policy
        env.step(action)
        image = env.render(args.player)
        recorded_frames.append(image)

    env.close()

    # Save video
    imageio.mimsave(args.output_path, recorded_frames, fps=args.output_fps)


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
    return parser.parse_args()


if __name__ == "__main__":
    opts = get_args()
    main(opts)
