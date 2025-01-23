import cairosvg
import chess
import chess.svg
import imageio
from pettingzoo.utils.env import ActionType, AgentID, ObsType
from pettingzoo.utils.wrappers.base import BaseWrapper


class RenderingWrapper(BaseWrapper[AgentID, ObsType, ActionType]):
    def render(self, size: int = 384):
        """render.

        Print the current game state.
        """
        if not hasattr(self, "game_state"):
            raise UserWarning(
                "You must reset the environment using reset() before calling render()."
            )

        board = chess.Board(str(self.game_state))
        svg_path = chess.svg.board(board, size=size)
        png_bytes = cairosvg.svg2png(svg_path)

        image = imageio.imread(png_bytes, format="png")

        return image
