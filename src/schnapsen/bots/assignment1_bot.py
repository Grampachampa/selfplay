from schnapsen.game import Bot, PlayerPerspective, Move, SchnapsenTrickScorer, SchnapsenGamePlayEngine
from schnapsen.bots.gui import SchnapsenServer
from schnapsen.bots.rand import RandBot
from typing import Optional
import random


engine = SchnapsenGamePlayEngine()

class BullyBot(Bot):
    """Bullybot in accordance with the rules of the assignment"""

    def __init__(self):
        super().__init__()

    def get_move(self, state: PlayerPerspective, leader_move: Optional[Move]):
        moves = state.valid_moves()
        trump_suit = state.get_trump_suit()
        enemy_move_suit = None
        if leader_move != None:
            enemy_move_suit = leader_move._cards()[0].suit
        trump_moves = []
        response_moves = []
        
        is_leader = state.am_i_leader()

        
        for move in moves:
            if move._cards()[0].suit == trump_suit:
                trump_moves.append(move)

            if move._cards()[0].suit == enemy_move_suit:
                response_moves.append(move)
        
        
        if len(trump_moves) > 0:
            return random.choice(trump_moves)

        elif is_leader == False and len(response_moves) > 0:
            return random.choice(response_moves)

        else:
            return sorted(moves, key = lambda a : SchnapsenTrickScorer().rank_to_points(a._cards()[0].rank), reverse = True)[0]


# Second Bot        
class SecondBot(Bot):
    """Second Bot in accordance to the assignment"""

    def __init__(self):
        super().__init__()

    def get_move(self, state: PlayerPerspective, leader_move: Optional[Move]):
        
        # valid moves
        moves = state.valid_moves()

        # my and enemy direct points
        my_score = state.get_my_score().direct_points
        enemy_score = state.get_opponent_score().direct_points

        # list of moves that have the same suit as my last move
        last_suit_moves = []
        current_hand = [i for i in state.get_hand()]
        
        # Get last move's suit if there has been a suit
        if len(state.get_game_history()) > 1:            
            old_hand = [i for i in state.get_game_history()[-2][0].get_hand()]
            number = 2
            last_suit = None

            while old_hand == current_hand and number <= len(state.get_game_history()):
                old_hand = [i for i in state.get_game_history()[-number][0].get_hand()]
                #print(old_hand)
                number +=1



            
            for card in old_hand:
                if card not in current_hand:
                    last_suit = card.suit
                    #print(last_suit)
                    break
                
            # makes oa list of moves that have the same suit as the previous card played
            for move in moves:
                if move._cards()[0].suit == last_suit:
                    last_suit_moves.append(move)


        # If bot has lower score than enemy, it looks for a marrige or trump exchange
        if my_score < enemy_score:
            for move in moves:
                if move.is_marriage() or move.is_trump_exchange():
                    chosen_move = move
                    #print(chosen_move)
                    return chosen_move
        
        # If a move of the previous suit is available, it does that next
        if len(last_suit_moves) > 0:
            chosen_move = sorted(last_suit_moves, key = lambda a : SchnapsenTrickScorer().rank_to_points(a._cards()[0].rank))[0]
            return chosen_move

        # Random non-marrige/non-exchange move
        else:
            chosen_move = random.choice([i for i in moves if i.is_marriage() == False and i.is_trump_exchange() == False])
            #print(chosen_move)
            return chosen_move





# My Children
bot1 = BullyBot()
bot2 = SecondBot()

# Bot VS Bot
for i in range(10):
    print (f"GAME {i+1}")
    result = engine.play_game(bot1, bot2, random.Random(i**2))
    print (f"Results:\nWinner: {result[0]}\nPoints: {result[1]}\nScore: {result[2]}\n\n-------------------\n")


# Man VS Machine
with SchnapsenServer() as s:
    player = s.make_gui_bot(name="BullyBot")
    engine.play_game(bot1, player, random.Random(100))
