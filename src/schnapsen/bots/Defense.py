from typing import Optional
from schnapsen.game import Bot, PlayerPerspective, Move, SchnapsenGamePlayEngine, SchnapsenTrickScorer, GamePhase
from schnapsen.bots.exercise1 import MyBot
from schnapsen.bots.tibot import Tibor
import random


class Defense(Bot):
    def __init__(self):
        pass

    def get_move(self, state: 'PlayerPerspective', leader_move: Optional['Move']) -> 'Move':
        '''
        If the opponent plays a non-trump card, play a higher card of the same suit
        else if a lower card of the same suit
        else if play a trump
        else play any card

        If the opponenet plays a trump card, play a higher trump
        else if play a lower trump
        else if play anything
        '''
        moves = state.valid_moves()
        chosen_move = moves[0]

        if not state.am_i_leader() and not leader_move.is_trump_exchange():
            for each_move in moves:
                # if the move is of the same suit
                if each_move.cards[0].suit == leader_move.cards[0].suit:
                    # if the rank of the card is higher in the same suit
                    if SchnapsenTrickScorer.rank_to_points(self, each_move.cards[0].rank) > SchnapsenTrickScorer.rank_to_points(self, leader_move.cards[0].rank):
                        #choose a move which is of the same suit and higher
                        chosen_move = each_move
                    # else if play a lower card of the same suit
                    elif SchnapsenTrickScorer.rank_to_points(self, each_move.cards[0].rank) < SchnapsenTrickScorer.rank_to_points(self, leader_move.cards[0].rank):
                        chosen_move = each_move
                # else if play a trump card
                elif each_move.is_trump_exchange():
                    chosen_move = each_move

                # else if play a random card
                else:
                    chosen_move = random.choice(moves)
        print('DefensiveBot', chosen_move, state.get_my_score().direct_points)
        return chosen_move


engine = SchnapsenGamePlayEngine()
bot1 = Trump()
bot2 = Tibor()
# bot1 = s.make_gui_bot(name="mybot1")
for i in range(10):
    print('New Game')
    engine.play_game(bot1, bot2, random.Random(i))