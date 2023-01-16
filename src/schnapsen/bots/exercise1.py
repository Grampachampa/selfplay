import random
from typing import Optional
from schnapsen.game import Bot, PlayerPerspective, Move, SchnapsenGamePlayEngine


class MyBot(Bot):
    '''
    MyBot plays in a way if my score is lower than the opponent's score, it will, play a trump exchange or a marriage card. Else if,
    it plays the same suit that it played in the turn before. Finally, if none of that is available, it plays a random move which
    is not a marriage or trump card.
    '''

    def __init__(self):
        self.previous_move = False

    def get_move(self, state: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        # get all possible valid moves
        moves = state.valid_moves()
        chosen_move = moves[0]

        # find my score in the game
        my_score = state.get_my_score().direct_points

        # find the opponents score in the game
        opponent_score = state.get_opponent_score().direct_points

        if my_score < opponent_score:

            for each_move in moves:
                # play the first trump card if it is available
                if each_move.is_trump_exchange():
                    chosen_move = each_move
                # play a marriage card if trump card is not available
                elif each_move.is_marriage():
                    chosen_move = each_move


        # play the card of the same suit if neither trump card and marriage card is not available
        elif self.previous_move:
            moves_same_suit = []
            for each_move in moves:
                if self.previous_move.cards[0].suit == each_move.cards[0].suit:
                    moves_same_suit.append(each_move)

            if len(moves_same_suit) > 0:
                chosen_move = random.choice(moves_same_suit)

        # play a random card if trump, marriage or the card of the same suit is not available
        else:
            random_moves = []
            moves = state.valid_moves()
            for each_move in moves:
                if not each_move.is_trump_exchange and not each_move.is_marriage:
                    random_moves.append(each_move)

            if len(random_moves) > 0:
                chosen_move = random.choice(random_moves)

        self.previous_move = chosen_move
        print('MyBot', chosen_move, my_score)
        return chosen_move
