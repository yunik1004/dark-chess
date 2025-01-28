from typing import Tuple
import xml.etree.ElementTree as ET
import cairosvg
import chess
import chess.svg
import imageio
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
from open_spiel.python.observation import make_observation
import pyspiel
from shimmy import OpenSpielCompatibilityV0


HIDDEN_SQUARE_SVG = """<g id="hidden"><path d="M0 0 L0 45 L45 45 L45 0 Z"
fill="#000" fill-opacity="0.60" /></g>"""


class DarkChessGame(OpenSpielCompatibilityV0):
    def __init__(
        self,
        cheat_mode: Tuple[bool | None, bool | None] = (False, False),
        render_mode: str | None = None,
    ):
        super().__init__(None, "dark_chess", render_mode, None)
        self.cheat_mode = cheat_mode

    def _update_action_spaces(self):
        for agent in self.possible_agents:
            try:
                self.action_spaces[agent] = spaces.Discrete(
                    self._env.num_distinct_actions(),
                    seed=self.np_random if hasattr(self, "np_random") else None,
                )
            except pyspiel.SpielError as e:
                raise NotImplementedError(
                    f"{str(e)[:-1]} for action space for {self._env}."
                )

    def render(self, agent_id: int = -1, size: int = 384) -> np.ndarray:
        """render the current game state."""
        if not hasattr(self, "game_state"):
            raise UserWarning(
                "You must reset the environment using reset() before calling render()."
            )

        # Chess board
        board = chess.Board(str(self.game_state))
        svg = chess.svg.board(board, size=size)

        if agent_id < 0:
            agent_id = self.game_state.current_player()
        observation = make_observation(self.game_state.get_game())

        # Draw board status
        svg = ET.fromstring(svg)
        defs = ET.SubElement(svg, "defs")

        # Draw fog
        defs.append(ET.fromstring(HIDDEN_SQUARE_SVG))
        if agent_id in self.agent_ids:
            observation.set_from(self.game_state, agent_id)

            public_obs = observation.dict["public_empty_pieces"]  # np.array
            for key, obs in observation.dict.items():
                if key.startswith("public_"):
                    public_obs = np.max((public_obs, obs), axis=0)
            # public_squares = np.where(public_obs.T.flatten())[0].tolist()

            unknown_obs = [
                v for k, v in observation.dict.items() if k.endswith("unknown_squares")
            ][0]
            hidden_obs = unknown_obs - public_obs
            hidden_squares = np.where(hidden_obs.T.flatten())[0].tolist()

            hidden_set = chess.SquareSet(hidden_squares)

            orientation = True
            margin = 15

            for square, bb in enumerate(chess.BB_SQUARES):
                file_index = chess.square_file(square)
                rank_index = chess.square_rank(square)
                x = (
                    file_index if orientation else 7 - file_index
                ) * chess.svg.SQUARE_SIZE + margin
                y = (
                    7 - rank_index if orientation else rank_index
                ) * chess.svg.SQUARE_SIZE + margin

                if square in hidden_set:
                    ET.SubElement(
                        svg,
                        "ns0:use",
                        chess.svg._attrs(
                            {
                                "href": "#hidden",
                                "x": x,
                                "y": y,
                            }
                        ),
                    )

        svg = chess.svg.SvgWrapper(ET.tostring(svg).decode("utf-8"))
        png_bytes = cairosvg.svg2png(svg)
        image = imageio.imread(png_bytes, format="png")

        return image

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ):
        """reset.

        Args:
            seed (Optional[int]): seed
            options (Optional[Dict]): options
        """
        # initialize np random the seed
        self.np_random, self.np_seed = seeding.np_random(seed)

        self.game_name = self.game_type.short_name

        # seed argument is only valid for three games
        if self.game_name in ["deep_sea", "hanabi", "mfg_garnet"] and seed is not None:
            if self.config is not None:
                reset_config = self.config.copy()
                reset_config["seed"] = seed
            else:
                reset_config = {"seed": seed}
            self._env = pyspiel.load_game(self.game_name, reset_config)
        else:
            if self.config is not None:
                self._env = pyspiel.load_game(self.game_name, self.config)
            else:
                self._env = pyspiel.load_game(self.game_name)

        # all agents
        self.agents = self.possible_agents[:]

        # Set cheater
        self.is_cheater = {
            self.agents[i]: self.np_random.choice([False, True])
            if self.cheat_mode[i] is None
            else self.cheat_mode[i]
            for i in self.agent_ids
        }

        # boilerplate stuff
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        # get a new game state, game_length = number of game nodes
        self.game_length = 1
        self.game_state = self._env.new_initial_state()

        # holders in case of simultaneous actions
        self.simultaneous_actions = dict()

        # make sure observation and action spaces are correct for this environment config
        self._update_observation_spaces()
        self._update_action_spaces()

        # step through chance nodes
        # then update obs and act masks
        # then choose next agent
        self._execute_chance_node()
        self._update_action_masks()
        self._update_observations()
        self._choose_next_agent()

    def _update_observations(self):
        super()._update_observations()

        """
        Order: public -> repetition -> side_to_play -> irreversible move -> private -> castling
        (13 +  // public boards:  piece types * colours + empty
         14)   // private boards: piece types * colours + empty + unknown
            * board_size_ * board_size_ +
        3 +    // public: repetitions count, one-hot encoding
        2 +    // public: side to play
        1 +    // public: irreversible move counter -- a fraction of $n over 100
        2 * 2  // private: left/right castling rights, one-hot encoded.
        """

        private_boards = {
            agent_id: np.reshape(self.observations[agent_name][-900:-4], (14, 8, 8))
            for agent_id, agent_name in zip(self.agent_ids, self.agents)
        }

        private_indices = {
            0: [1, 3, 5, 7, 9, 11],
            1: [0, 2, 4, 6, 8, 10],
        }

        # Modify private board of the cheat player
        for agent_id, agent_name in zip(self.agent_ids, self.agents):
            if self.is_cheater[agent_name]:
                public_board = np.reshape(
                    self.observations[agent_name][:832], (13, 8, 8)
                )

                opposite_id = 1 - agent_id
                private_self = private_boards[agent_id][private_indices[agent_id]]
                private_opposite = private_boards[1 - agent_id][
                    private_indices[opposite_id]
                ]

                private_board_new = np.zeros((14, 8, 8), dtype=np.float64)
                private_board_new[private_indices[agent_id]] = private_self
                private_board_new[private_indices[opposite_id]] = private_opposite

                empty_board = 1.0 - np.vstack([public_board, private_board_new]).max(
                    axis=0
                )
                private_board_new[-2] = empty_board

                self.observations[agent_name][-900:-4] = np.reshape(
                    private_board_new, (-1)
                ).tolist()
