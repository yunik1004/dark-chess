from typing import Optional
from shimmy import OpenSpielCompatibilityV0


def make_dark_chess_env(render_mode: Optional[str] = None):
    return OpenSpielCompatibilityV0(game_name="dark_chess", render_mode=render_mode)
