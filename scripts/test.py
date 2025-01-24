import imageio
from dark_chess.envs import make_dark_chess_env
from dark_chess.envs.wrappers import RenderingWrapper


env = make_dark_chess_env()
env = RenderingWrapper(env)

env.reset()
obs = []
frames = []

player = -1

image = env.render(player)
frames.append(image)
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample(
            info["action_mask"]
        )  # this is where you would insert your policy
    env.step(action)
    obs.append(observation)
    image = env.render(player)
    frames.append(image)
env.close()

imageio.mimsave("out.mp4", frames, fps=2)
