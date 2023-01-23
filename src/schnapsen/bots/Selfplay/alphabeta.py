import random
from typing import Optional

from schnapsen.game import Bot, Move, PlayerPerspective


class AlphaBetaBot(Bot):

    __max_depth = 1
    __randomise = True

    def __init__(self, randomise=True, depth=0) -> None:
        self.__randomise = randomise
        self.__max_depth = depth


    def get_move(self, player_perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        val, move = self.value(player_perspective)

        return move

    def value(self, state, alpha=float('-inf'), beta=float('inf'), depth=0):

        if state.get_my_score().pending_points == 0 or state.get_opponent_score().pending_points == 0:
            winner = None
            points = None

            if state.get_my_score().direct_points >= 66:
                winner = 1
            elif state.get_opponent_score().direct_points >= 66:
                winner = 2

            if winner == 1:
                if state.get_opponent_score().direct_points == 0:
                    points = 3
                elif state.get_opponent_score().direct_points < 33:
                    points = 2
                else:
                    points = 1
            else:
                if state.get_my_score().direct_points == 0:
                    points = 3
                elif state.get_my_score().direct_points < 33:
                    points = 2
                else:
                    points = 1

            return (points, None) if winner == 1 else (-points, None)

        if depth == self.__max_depth:
            return heuristic(state)

        best_value = float('-inf') if maximizing(state) else float('-inf')
        best_move = None

        moves = state.valid_moves()

        if self.__randomise:
            random.shuffle(moves)

        for move in moves:

            next_state = state.get_state_in_phase_two(move)
            value, _ = self.value(next_state)

            if maximizing(state):
                if value > best_value:
                    best_value = value
                    best_move = move
                    alpha = best_value
            else:
                if value < best_value:
                    best_value = value
                    best_move = move
                    beta = best_value

            if alpha >= beta:
                break
        return best_value, best_move


def maximizing(state):

    if state.am_i_leader():
        return 1


def heuristic(state):
    if state.get_my_score().direct_points + state.get_opponent_score().direct_points != 0:
        return state.get_my_score().direct_points / float((state.get_my_score().direct_points + state.get_opponent_score().direct_points))
    return 0