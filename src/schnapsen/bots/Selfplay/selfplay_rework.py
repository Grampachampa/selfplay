from schnapsen.game import *
from schnapsen.bots import RandBot, RdeepBot #.rand import RandBot
from typing import Optional
import random
from random import Random
from typing import Iterable, List, Optional, Tuple, Union, cast, Any
import torch
import os
from collections import deque
from model import Linear_QNet, QTrainer
from helper import plot
import alphabeta




# TODO: silence plot, change while true to a certain number of iterations, save only last iter
# TODO: CPU-> GPU
# TODO: install pytorch on DAS
# TODO: Figure out where to save files

class SelfPlay (Bot):
    """Self-play reinforcement learning schnapsen god of destruction"""

    MAX_MEMORY = 100_000
    BATCH_SIZE = 10000
    LR = 0.01

    def __init__(self) -> None:
        self.number_of_games = 0
        self.trick_number = 0
        self.max_epsilon = 1000
        self.epsilon = 1001
        self.min_epsilon = 10
        self.epsilon_decay = 1
        self.gamma = 0.8 # discount rate < 1
        self.memory = deque(maxlen=self.MAX_MEMORY)
        self.model = Linear_QNet(173, 256, 8)
        self.trainer = QTrainer(self.model, lr = self.LR, gamma = self.gamma)
        self.my_match_points = 7
        self.opponent_match_points = 7

        self.current_state0 = None
        self.next_state0 = None
        self.move0 = [0,0,0,0,0,0,0,0]

        self.reward = 0
        self.done = False

    def save(self):
        #print("Saved!")
        self.model.save(iter=self.number_of_games, file_name = 'snapshot.pth')



        
    def get_move(self, player_perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:

        
        self.trick_number += 1

        # define epsilon based on experience
        if self.epsilon > self.min_epsilon: 
            self.epsilon = self.max_epsilon - (self.number_of_games * self.epsilon_decay)
        else: self.epsilon = self.min_epsilon
        
        # get Valid moves
        state = player_perspective
     
        #print(state.get_my_score().direct_points, state.get_opponent_score().direct_points)
        
        state_representation = create_state_and_actions_vector_representation(state, leader_move=leader_move, follower_move=None)
        state0 = torch.tensor(create_state_and_actions_vector_representation(state, leader_move=leader_move, follower_move=None), device="cuda")
        moves = state.valid_moves()

        self.next_state0 = state_representation
        
        self.reward = (state.get_my_score().direct_points - state.get_opponent_score().direct_points)/self.trick_number

        if self.trick_number != 1:
            self.train_short_memory(self.current_state0, self.move0, self.reward, self.next_state0, self.done)
            self.remember(self.current_state0, self.move0, self.reward, self.next_state0, self.done)


        # ensure that valid moves are alwyas in a predictable order
        moves = self.move_order(moves)
        


        # check if it's time for some good ol' fashioned AB pruning - if so, prune with extreme predjudice
        phase = state.get_phase()


        '''
        if phase == GamePhase.TWO:
            ab = alphabeta.AlphaBetaBot()
            final_move = ab.get_move(player_perspective, leader_move)
            move_index = moves.index(final_move)
            '''




        
        # for first 50 gen, it will make random moves sometimes, just to get more data
        if random.randint(0,self.max_epsilon) < self.epsilon:
            move_index = random.randint(0,len(moves)-1)
            true_move_index = move_index
            final_move = moves[move_index]
            #print("(", end="")


        # otherwise, it gets spicy
        else:
            # get prediction from model, convert it to index in list of moves, and return final move
            prediction = self.model(state0)
            
            move_index = torch.argmax(prediction).item()
            true_move_index = move_index
            
            
            while move_index > len(moves)-1:
                move_index -= len(moves)

            final_move = moves[move_index]
        
        self.move0 = [0,0,0,0,0,0,0,0]
        self.move0[true_move_index] = 1

        #print(move_index, end="")
        self.current_state0 = state_representation


            
        return final_move


    def remember(self, state, move_index, reward, next_state, done):
        self.memory.append((state, move_index, reward, next_state, done))

    # train long memory
    def train_long_memory(self):
        if len(self.memory) > self.BATCH_SIZE:
            mini_sample = random.sample(self.memory, self.BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory
        

        states, actions, rewards, next_states, dones = zip(*mini_sample)

        self.trainer.train_step(states, actions, rewards, next_states, dones)

        
    # Short Memory; works like a charm
    def train_short_memory(self, state, move_index, reward, next_state, done):
        self.trainer.train_step(state, move_index, reward, next_state, done)
    
    # order moves by move type, then suit, then rank (alphabetically, of course. How else would we do it?)
    def move_order (self, moves: List[Move]):
        regular_moves: List[RegularMove] = [i for i in moves if type(i) == RegularMove]
        trump_moves: List[Trump_Exchange] = [i for i in moves if type(i) == Trump_Exchange]
        marriage_moves: List[Marriage] = [i for i in moves if type(i) == Marriage]
        
        regular_moves.sort(key= lambda a: (str(a._cards()[0].suit), str(a._cards()[0].value[0])))
        trump_moves.sort(key= lambda a: (str(a._cards()[0].suit), str(a._cards()[0].value[0])))
        marriage_moves.sort(key= lambda a: (str(a._cards()[0].suit), str(a._cards()[0].value[0])))

        return regular_moves + trump_moves + marriage_moves



class TrainingEngine(SchnapsenGamePlayEngine):

    def play_game(self, bot1: Bot, bot2: Bot, rng: Random) -> Tuple[Bot, int, Score]:
        """
        Play a game between bot1 and bot2, using the rng to create the game.

        :param bot1: The first bot playing the game. This bot will be the leader for the first trick.
        :param bot2: The second bot playing the game. This bot will be the follower for the first trick.
        :param rng: The random number generator used to shuffle the deck.

        :returns: A tuple with the bot which won the game, the number of points obtained from this game and the score attained.
        """
        cards = self.deck_generator.get_initial_deck()
        shuffled = self.deck_generator.shuffle_deck(cards, rng)
        hand1, hand2, talon = self.hand_generator.generateHands(shuffled)

        leader_state = BotState(implementation=bot1, hand=hand1)
        follower_state = BotState(implementation=bot2, hand=hand2)

        game_state = GameState(
            leader=leader_state,
            follower=follower_state,
            talon=talon,
            previous=None
        )
        winner, points, score, winner_state, loser_state = self.play_game_from_state(game_state=game_state, leader_move=None)

        
        return winner, points, score, winner_state, loser_state



    def play_game_from_state(self, game_state: GameState, leader_move: Optional[Move]) -> Tuple[Bot, int, Score]:
        """
        Continue a game  which might have been started before.
        The leader move is an optional paramter which can be provided to force this first move from the leader.

        :param game_state: The state of the game to start from
        :param leader_move: if provided, the leader will be forced to play this move as its first move.

        :returns: A tuple with the bot which won the game, the number of points obtained from this game and the score attained.
        """
        winner: Optional[BotState] = None
        points: int = -1
        while not winner:
            if leader_move is not None:
                # we continues from a game where the leading bot already did a move, we immitate that
                game_state = self.trick_implementer.play_trick_with_fixed_leader_move(game_engine=self, game_state=game_state, leader_move=leader_move)
                leader_move = None
            else:
                game_state = self.trick_implementer.play_trick(self, game_state)
            winner, points = self.trick_scorer.declare_winner(game_state) or (None, -1)

        winner_state = WinnerPerspective(game_state, self)
        winner.implementation.notify_game_end(won=True, state=winner_state)

        loser_state = LoserPerspective(game_state, self)
        game_state.follower.implementation.notify_game_end(False, state=loser_state)

        return winner.implementation, points, winner.score, winner_state, loser_state



class ModelReader(Bot):

    def __init__(self, snapshot_path):
        self.model = Linear_QNet(173, 256, 8)
        self.model.load_state_dict(torch.load(snapshot_path))
        self.trick_number = 0

    def get_move(self, player_perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:

        self.trick_number +=1

        state = player_perspective             
        state0 = torch.tensor(create_state_and_actions_vector_representation(state, leader_move=leader_move, follower_move=None), device="cuda")
        moves = state.valid_moves()
        moves = self.move_order(moves)

                
        # get prediction from model, convert it to index in list of moves, and return final move
        prediction = self.model(state0)
            
        move_index = torch.argmax(prediction).item()
                        
        while move_index > len(moves)-1:
            move_index -= len(moves)

        final_move = moves[move_index]
           
        return final_move


    
    # order moves by move type, then suit, then rank (alphabetically, of course. How else would we do it?)
    def move_order (self, moves: List[Move]):
        regular_moves: List[RegularMove] = [i for i in moves if type(i) == RegularMove]
        trump_moves: List[Trump_Exchange] = [i for i in moves if type(i) == Trump_Exchange]
        marriage_moves: List[Marriage] = [i for i in moves if type(i) == Marriage]
        
        regular_moves.sort(key= lambda a: (str(a._cards()[0].suit), str(a._cards()[0].value[0])))
        trump_moves.sort(key= lambda a: (str(a._cards()[0].suit), str(a._cards()[0].value[0])))
        marriage_moves.sort(key= lambda a: (str(a._cards()[0].suit), str(a._cards()[0].value[0])))

        return regular_moves + trump_moves + marriage_moves

    




# shamelessly ripped from mlbot
def create_state_and_actions_vector_representation(player_perspective: PlayerPerspective, leader_move: Optional[Move], follower_move: Optional[Move]) -> List[int]:
    """
    This function takes as input a PlayerPerspective variable, and the two moves of leader and follower,
    and returns a list of complete feature representation that contains all information
    """
    player_game_state_representation = get_state_feature_vector(player_perspective)
    leader_move_representation = get_move_feature_vector(leader_move)
    follower_move_representation = get_move_feature_vector(follower_move)

    #print(player_game_state_representation , leader_move_representation , follower_move_representation)

    return player_game_state_representation + leader_move_representation + follower_move_representation

def get_one_hot_encoding_of_card_suit(card_suit: Suit) -> List[int]:
    """
    Translating the suit of a card into one hot vector encoding of size 4 and type of numpy ndarray.
    """
    card_suit_one_hot: list[int]
    if card_suit == Suit.HEARTS:
        card_suit_one_hot = [0, 0, 0, 1]
    elif card_suit == Suit.CLUBS:
        card_suit_one_hot = [0, 0, 1, 0]
    elif card_suit == Suit.SPADES:
        card_suit_one_hot = [0, 1, 0, 0]
    elif card_suit == Suit.DIAMONDS:
        card_suit_one_hot = [1, 0, 0, 0]
    else:
        raise ValueError("Suit of card was not found!")

    return card_suit_one_hot

def get_one_hot_encoding_of_card_rank(card_rank: Rank) -> List[int]:
    """
    Translating the rank of a card into one hot vector encoding of size 13 and type of numpy ndarray.
    """
    card_rank_one_hot: list[int]
    if card_rank == Rank.ACE:
        card_rank_one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    elif card_rank == Rank.TWO:
        card_rank_one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    elif card_rank == Rank.THREE:
        card_rank_one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    elif card_rank == Rank.FOUR:
        card_rank_one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    elif card_rank == Rank.FIVE:
        card_rank_one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    elif card_rank == Rank.SIX:
        card_rank_one_hot = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    elif card_rank == Rank.SEVEN:
        card_rank_one_hot = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    elif card_rank == Rank.EIGHT:
        card_rank_one_hot = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    elif card_rank == Rank.NINE:
        card_rank_one_hot = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    elif card_rank == Rank.TEN:
        card_rank_one_hot = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif card_rank == Rank.JACK:
        card_rank_one_hot = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif card_rank == Rank.QUEEN:
        card_rank_one_hot = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif card_rank == Rank.KING:
        card_rank_one_hot = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    else:
        raise AssertionError("Provided card Rank does not exist!")
    return card_rank_one_hot

def get_move_feature_vector(move: Optional[Move]) -> List[int]:
    """
        in case there isn't any move provided move to encode, we still need to create a "padding"-"meaningless" vector of the same size,
        filled with 0s, since the ML models need to receive input of the same dimensionality always.
        Otherwise, we create all the information of the move i) move type, ii) played card rank and iii) played card suit
        translate this information into one-hot vectors respectively, and concatenate these vectors into one move feature representation vector
    """

    if move is None:
        move_type_one_hot_encoding_numpy_array = [0, 0, 0]
        card_rank_one_hot_encoding_numpy_array = [0, 0, 0, 0]
        card_suit_one_hot_encoding_numpy_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    else:
        move_type_one_hot_encoding: list[int]
        # in case the move is a marriage move
        if move.is_marriage():
            move_type_one_hot_encoding = [0, 0, 1]
            card = move.queen_card
        #  in case the move is a trump exchange move
        elif move.is_trump_exchange():
            move_type_one_hot_encoding = [0, 1, 0]
            card = move.jack
        #  in case it is a regular move
        else:
            move_type_one_hot_encoding = [1, 0, 0]
            card = move.card
        move_type_one_hot_encoding_numpy_array = move_type_one_hot_encoding
        card_rank_one_hot_encoding_numpy_array = get_one_hot_encoding_of_card_rank(card.rank)
        card_suit_one_hot_encoding_numpy_array = get_one_hot_encoding_of_card_suit(card.suit)

    return move_type_one_hot_encoding_numpy_array + card_rank_one_hot_encoding_numpy_array + card_suit_one_hot_encoding_numpy_array

def get_state_feature_vector(player_perspective: PlayerPerspective) -> List[int]:
    """
        This function gathers all subjective information that this bot has access to, that can be used to decide its next move, including:
        - points of this player (int)
        - points of the opponent (int)
        - pending points of this player (int)
        - pending points of opponent (int)
        - the trump suit (1-hot encoding)
        - phase of game (1-hoy encoding)
        - talon size (int)
        - if this player is leader (1-hot encoding)
        - What is the status of each card of the deck (where it is, or if its location is unknown)

        Important: This function should not include the move of this agent.
        It should only include any earlier actions of other agents (so the action of the other agent in case that is the leader)
    """
    # a list of all the features that consist the state feature set, of type np.ndarray
    state_feature_list: list[int] = []

    player_score = player_perspective.get_my_score()
    # - points of this player (int)
    player_points = player_score.direct_points
    # - pending points of this player (int)
    player_pending_points = player_score.pending_points

    # add the features to the feature set
    state_feature_list += [player_points]
    state_feature_list += [player_pending_points]

    opponents_score = player_perspective.get_opponent_score()
    # - points of the opponent (int)
    opponents_points = opponents_score.direct_points
    # - pending points of opponent (int)
    opponents_pending_points = opponents_score.pending_points

    # add the features to the feature set
    state_feature_list += [opponents_points]
    state_feature_list += [opponents_pending_points]

    # - the trump suit (1-hot encoding)
    trump_suit = player_perspective.get_trump_suit()
    trump_suit_one_hot = get_one_hot_encoding_of_card_suit(trump_suit)
    # add this features to the feature set
    state_feature_list += trump_suit_one_hot

    # - phase of game (1-hot encoding)
    game_phase_encoded = [1, 0] if player_perspective.get_phase() == GamePhase.TWO else [0, 1]
    # add this features to the feature set
    state_feature_list += game_phase_encoded

    # - talon size (int)
    talon_size = player_perspective.get_talon_size()
    # add this features to the feature set
    state_feature_list += [talon_size]

    # - if this player is leader (1-hot encoding)
    i_am_leader = [0, 1] if player_perspective.am_i_leader() else [1, 0]
    # add this features to the feature set
    state_feature_list += i_am_leader

    # gather all known deck information
    hand_cards = player_perspective.get_hand().cards
    trump_card = player_perspective.get_trump_card()
    won_cards = player_perspective.get_won_cards().get_cards()
    opponent_won_cards = player_perspective.get_opponent_won_cards().get_cards()
    opponent_known_cards = player_perspective.get_known_cards_of_opponent_hand().get_cards()
    # each card can either be i) on player's hand, ii) on player's won cards, iii) on opponent's hand, iv) on opponent's won cards
    # v) be the trump card or vi) in an unknown position -> either on the talon or on the opponent's hand
    # There are all different cases regarding card's knowledge, and we represent these 6 cases using one hot encoding vectors as seen bellow.

    deck_knowledge_in_consecutive_one_hot_encodings: list[int] = []
    sdg = SchnapsenDeckGenerator()

    for card in sdg.get_initial_deck():
        card_knowledge_in_one_hot_encoding: list[int]
        # i) on player's hand
        if card in hand_cards:
            card_knowledge_in_one_hot_encoding = [0, 0, 0, 0, 0, 1]
        # ii) on player's won cards
        elif card in won_cards:
            card_knowledge_in_one_hot_encoding = [0, 0, 0, 0, 1, 0]
        # iii) on opponent's hand
        elif card in opponent_known_cards:
            card_knowledge_in_one_hot_encoding = [0, 0, 0, 1, 0, 0]
        # iv) on opponent's won cards
        elif card in opponent_won_cards:
            card_knowledge_in_one_hot_encoding = [0, 0, 1, 0, 0, 0]
        # v) be the trump card
        elif card == trump_card:
            card_knowledge_in_one_hot_encoding = [0, 1, 0, 0, 0, 0]
        # vi) in an unknown position as it is invisible to this player. Thus, it is either on the talon or on the opponent's hand
        else:
            card_knowledge_in_one_hot_encoding = [1, 0, 0, 0, 0, 0]
        # This list eventually develops to one long 1-dimensional numpy array of shape (120,)
        deck_knowledge_in_consecutive_one_hot_encodings += card_knowledge_in_one_hot_encoding
    # deck_knowledge_flattened: np.ndarray = np.concatenate(tuple(deck_knowledge_in_one_hot_encoding), axis=0)

    # add this features to the feature set
    state_feature_list += deck_knowledge_in_consecutive_one_hot_encodings

    return state_feature_list


def train():

    engine = TrainingEngine() 
    rng = random.Random(12)
    main_bot = SelfPlay()
    opponents = (RandBot(rng), RdeepBot(num_samples=4, depth=8, rand=rng))
    
    plot_winrate_last_n, n = [], 50
    last_n = []
    plot_winrate_all_time = []
    mywins = 0
    
    dirname = os.path.dirname(__file__)


    
    # Training all happens within the following while loop:
    for i in range(100000):

        if main_bot.number_of_games < 51:
            adversary_bot = opponents[0]
        
        else:
            iter = random.choice([i for i in range(1,main_bot.number_of_games) if not i%50])
            path = f'./selfplay_snapshots/generation{iter}_snapshot.pth'
            path = os.path.join(dirname, path)
            adversary_bot = ModelReader(path)

        players = [main_bot, adversary_bot]
        random.shuffle(players)
        winner, points, score, winner_state, loser_state = engine.play_game(players[0], players[1], rng)


        main_bot.number_of_games += 1

        if winner == main_bot:
            main_bot.next_state0 = create_state_and_actions_vector_representation(winner_state, None, None)
            main_bot.reward = (winner_state.get_my_score().direct_points - loser_state.get_my_score().direct_points + 60)/(main_bot.trick_number+1)
            winner_declaration = True
            mywins += 1
            #print("\n", "True", end="")

            
        else:
            main_bot.next_state0 = create_state_and_actions_vector_representation(loser_state, None, None)
            main_bot.reward = (loser_state.get_my_score().direct_points - winner_state.get_my_score().direct_points - 60)/(main_bot.trick_number+1)
            winner_declaration = False
            print("\n", "False", end="")

        #print(main_bot.next_state0)
        main_bot.train_short_memory(main_bot.current_state0, main_bot.move0, main_bot.reward, main_bot.next_state0, True)
        main_bot.remember(main_bot.current_state0, main_bot.move0, main_bot.reward, main_bot.next_state0, True)
        main_bot.train_long_memory()
        

        main_bot.trick_number = 0

        if not main_bot.number_of_games % 50:
            main_bot.save()
        
        '''
        print(points)


        if main_bot.number_of_games <= n:
                last_n.append(winner_declaration)
                tracking_length = main_bot.number_of_games
        else:
            del last_n[0]
            last_n.append(winner_declaration)
            tracking_length = n
            

            
        plot_winrate_all_time.append(mywins/main_bot.number_of_games)
        plot_winrate_last_n.append(len([i for i in last_n if i == True])/tracking_length)

        # one line represents winrate over last 50 games, other line represents total winrate vs opponent
        plot( 
            plot_winrate_all_time, 
            plot_winrate_last_n, 
          )'''

        

if __name__ == "__main__":
    train()

'''
self_learning_bot = 0
enemy = 0
path = './selfplay_snapshots/generation6550_snapshot.pth'
dirname = os.path.dirname(__file__)
path = os.path.join(dirname, path)

model = ModelReader(path)
for i in range(1000):
    path = './selfplay_snapshots/generation6550_snapshot.pth'
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, path)
    
    rando = RandBot(random.Random(i)) #RdeepBot(20, 5,random.Random(i))
    
    bots = [rando, model]
    random.shuffle(bots)
    rng = random.Random(i**2)

    engine = SchnapsenGamePlayEngine()

    winner, points, score = engine.play_game(bots[0],bots[1], rng)

    if winner == model:
        self_learning_bot += 1
        #print("Model")
        continue
        
    enemy +=1
    #print("Enemy")

print(self_learning_bot, enemy)
print (model.trick_number/1000)'''