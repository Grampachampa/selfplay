import random
from typing import Optional
from schnapsen.game import Bot, PlayerPerspective, Move, SchnapsenTrickScorer

class Bully(Bot):
    '''
    This class is a bully bot. If the bully bot contains a Trump card, it will play that first. If not, then it will
    play a random card of the same suit that the other player played before. This can only happen if the bully bot plays second.
    Finally, if the card of the same suit is not available then, the bully bot plays the highest ranking card.
    '''
    def get_move(self, state: PlayerPerspective, leader_move: Optional[Move], ) -> Move:
        moves = state.valid_moves()
        chosen_move = moves[0]

        # a list with all the trump suits
        moves_trump_suit = []
        for each_move in moves:
            # if i have a trump exchange move
            if each_move.cards[0].suit == state.get_trump_suit():
                moves_trump_suit.append(each_move)

        # picking a random trump move from the list of trump suits
        if len(moves_trump_suit) > 0:
            chosen_move = random.choice(moves_trump_suit)


        # if i am the leader in the game
        elif not state.am_i_leader():
            # all possible valid moves of the player
            moves_same_suit = []
            for each_move in moves:
                if leader_move.cards[0].suit == each_move.cards[0].suit:
                    moves_same_suit.append(each_move)

            if len(moves_same_suit) > 0:
                chosen_move = random.choice(moves_same_suit)


        else:
            # returning the highest ranked move
            maximum_rank = 0
            for each_move in moves:
                if SchnapsenTrickScorer.rank_to_points(self, each_move.cards[0].rank) > maximum_rank:
                    maximum_rank = SchnapsenTrickScorer.rank_to_points(self, each_move.cards[0].rank)
                    chosen_move = each_move

        print('Bully', chosen_move, state.get_my_score().direct_points)
        return chosen_move