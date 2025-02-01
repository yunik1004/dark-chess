import chess
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
                    shape=(self.unwrapped._env.num_distinct_actions(),),
                    dtype=np.int8,
                ),
            }
        )


class RewardShapingWrapper(BaseWrapper[AgentID, ObsType, ActionType]):
    def __init__(
        self,
        env: AECEnv[AgentID, ObsType, ActionType],
        reward_piece: float = 0.01,
        reward_check: float = 0.1,
    ) -> None:
        super().__init__(env)
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 5,
            chess.ROOK: 7,
            chess.QUEEN: 9,
        }
        self.reward_piece = reward_piece
        self.reward_check = reward_check

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed, options=options)
        self.status_prev = self.get_status()

    def step(self, action: ActionType) -> None:
        super().step(action)
        status_curr = self.get_status()

        rewards_aux = {color: 0.0 for color in chess.COLORS}
        for color in chess.COLORS:
            if status_curr["num_colors"][color] < self.status_prev["num_colors"][color]:
                for piece in chess.PIECE_TYPES:
                    if (
                        piece != chess.KING
                        and status_curr["num_pieces"][piece]
                        < self.status_prev["num_pieces"][piece]
                    ):
                        # piece with color is captured
                        rewards_aux[color] -= (
                            self.piece_values[piece] * self.reward_piece
                        )
                        rewards_aux[not color] += (
                            self.piece_values[piece] * self.reward_piece
                        )
                        break

            if status_curr["is_check"][color]:
                rewards_aux[not color] += self.reward_check

        for color in chess.COLORS:
            agent_id = f"player_{int(color)}"
            if agent_id in self.rewards:
                self.rewards[agent_id] += rewards_aux[color]
                self._cumulative_rewards[agent_id] += rewards_aux[color]

    def get_status(self) -> dict:
        board = chess.Board(str(self.game_state))

        num_colors = {}
        for color in chess.COLORS:
            num_colors[color] = board.occupied_co[color].bit_count()

        num_pieces = {}
        for piece in chess.PIECE_TYPES:
            if piece == chess.KING:
                continue
            num_pieces[piece] = getattr(
                board, f"{chess.PIECE_NAMES[piece]}s"
            ).bit_count()

        is_check = {color: False for color in chess.COLORS}
        is_check[board.turn] = board.is_check()

        return {
            "num_colors": num_colors,
            "num_pieces": num_pieces,
            "is_check": is_check,
        }


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
