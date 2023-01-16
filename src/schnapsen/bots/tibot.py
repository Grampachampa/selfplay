from typing import Optional
from schnapsen.game import Bot, PlayerPerspective, Move, SchnapsenGamePlayEngine, SchnapsenTrickScorer, GamePhase
from schnapsen.bots import RandBot
from schnapsen.
import random


class Tibor(Bot):
    '''
    This class is a bully bot. If the bully bot contains a Trump card, it will play that first. If not, then it will
    play a random card of the same suit that the other player played before. This can only happen if the bully bot plays second.
    Finally, if the card of the same suit is not available then, the bully bot plays the highest ranking card.
    '''
    def get_move(self, state: PlayerPerspective, leader_move: Optional[Move], ) -> Move:
        moves = state.valid_moves()
        chosen_move = moves[0]

        # once the stock is exhausted
        if state.get_phase == GamePhase.TWO:
            trump_moves = []

            for each_move in moves:
                # playing trump cards to gain trump control
                if state.get_trump_suit() == each_move.cards[0].suit:
                    trump_moves.append(each_move)
            # choosing a random card from the trump cards
            if len(trump_moves) > 0:
                chosen_move = random.choice(trump_moves)
            else:
                chosen_move = random.choice(moves)


        #if the bot is playing the leader move
        else:
            if state.am_i_leader():
                highest_rank = 0
                highest_ranking_move = None
                for each_move in moves:

                    #playing non trump ace and ten ranking cards
                    if state.get_trump_suit() != each_move.cards[0].suit and each_move.cards[0].rank.ACE or each_move.cards[0].rank.TEN:
                        #choosing the highest ranking among the specified variants above
                        if SchnapsenTrickScorer.rank_to_points(self, each_move.cards[0].rank) > highest_rank:
                            highest_rank = SchnapsenTrickScorer.rank_to_points(self, each_move.cards[0].rank)
                            highest_ranking_move = each_move
                chosen_move = highest_ranking_move

            #if a am the follower
            else:
                highest_rank = 0
                #choose a random move if there is no same suit of cards
                highest_ranking_move = random.choice(moves)
                for each_move in moves:

                    #plating the card of the same suit
                    if each_move.cards[0].suit == leader_move.cards[0].suit:
                        # choosing the highest ranking among the specified variants above
                        if SchnapsenTrickScorer.rank_to_points(self, each_move.cards[0].rank) > highest_rank:
                            highest_rank = SchnapsenTrickScorer.rank_to_points(self, each_move.cards[0].rank)
                            highest_ranking_move = each_move
                chosen_move = highest_ranking_move


        print('Tibot', chosen_move, state.get_my_score().direct_points)

        return chosen_move


engine = SchnapsenGamePlayEngine()
bot1 = Tibor()
bot2 = ()
# bot1 = s.make_gui_bot(name="mybot1")
for i in range(10):
    print('New Game')
    engine.play_game(bot1, bot2, random.Random(2))