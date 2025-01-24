import xml.etree.ElementTree as ET
import cairosvg
import chess
import chess.svg
import imageio
import numpy as np
from open_spiel.python.observation import make_observation
from pettingzoo.utils.env import ActionType, AgentID, ObsType
from pettingzoo.utils.wrappers.base import BaseWrapper


HIDDEN_SQUARE_SVG = """<g id="hidden"><path d="M0 0 L0 45 L45 45 L45 0 Z"
fill="#000" fill-opacity="0.60" /></g>"""


class RenderingWrapper(BaseWrapper[AgentID, ObsType, ActionType]):
    def render(self, player: int = -1, size: int = 384) -> np.ndarray:
        """render the current game state."""
        if not hasattr(self, "game_state"):
            raise UserWarning(
                "You must reset the environment using reset() before calling render()."
            )

        # Chess board
        board = chess.Board(str(self.game_state))
        svg = chess.svg.board(board, size=size)

        if player < 0:
            player = self.game_state.current_player()
        observation = make_observation(self.game_state.get_game())

        # Draw board status
        svg = ET.fromstring(svg)
        defs = ET.SubElement(svg, "defs")

        # Draw fog
        defs.append(ET.fromstring(HIDDEN_SQUARE_SVG))
        if player in [0, 1]:
            observation.set_from(self.game_state, player)

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
