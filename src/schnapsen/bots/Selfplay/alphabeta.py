from schnapsen.game import *

class AlphaBeta(Bot):
    def __init__(self):
        super().__init__()

    def get_move(self, state: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        pass
