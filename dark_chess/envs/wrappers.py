from gymnasium import spaces
import imageio
import numpy as np
from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType
from pettingzoo.utils.wrappers.base import BaseWrapper


class ActionMask2ObservationWrapper(BaseWrapper[AgentID, ObsType, ActionType]):
    def observe(self, agent: AgentID) -> ObsType | None:
        return {
            "obs": self.env.observe(agent),
            "action_mask": self.env.infos[agent]["action_mask"],
        }

    def observation_space(self, agent: AgentID) -> spaces.Space:
        return spaces.Dict(
            {
                "obs": self.env.observation_space(agent),
                "action_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.env._env.num_distinct_actions(),),
                    dtype=np.int8,
                ),
            }
        )


class VideoRenderingWrapper(BaseWrapper[AgentID, ObsType, ActionType]):
    def __init__(
        self,
        env: AECEnv[AgentID, ObsType, ActionType],
        agent_id: int,
        video_path: str,
        fps: int,
    ) -> None:
        super().__init__(env)
        self.agent_id = agent_id
        self.video_path = video_path
        self.fps = fps
        self.frames = []

    def close(self) -> None:
        super().close()
        imageio.mimsave(self.video_path, self.frames, fps=self.fps)

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed, options=options)
        self.frames = []
        image = self.env.render(self.agent_id)
        self.frames.append(image)

    def step(self, action: ActionType) -> None:
        super().step(action)
        image = self.env.render(self.agent_id)
        self.frames.append(image)
