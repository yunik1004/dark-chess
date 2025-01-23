import imageio
from shimmy import OpenSpielCompatibilityV0
from dark_chess.envs.wrappers import RenderingWrapper

env = OpenSpielCompatibilityV0(game_name="dark_chess")
env = RenderingWrapper(env)

env.reset()
obs = []
frames = []
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
    image = env.render()
    frames.append(image)
env.close()

imageio.mimsave("out.mp4", frames, fps=2)
