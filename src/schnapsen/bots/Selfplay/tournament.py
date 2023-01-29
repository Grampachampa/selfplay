import sys
sys.path.insert(1, 'C:/Users/gramp/Desktop/Uni/IS Project/Self-Play bot project/src/schnapsen/bots')
from selfplay_rework import ModelReader
from schnapsen.bots import RandBot, RdeepBot
from tibot import *
from bully import *
from Defense import Defense
import random
from schnapsen.game import *
import json

engine = SchnapsenGamePlayEngine()
selfplay = ModelReader("C:/Users/gramp/Desktop/Uni/IS Project/Self-Play bot project/src/schnapsen/bots/Selfplay/selfplay_snapshots/generation10900_snapshot.pth")
result_dict = {}
for i in range(10000):
    opponents = {RandBot(random.Random(i)): "RandBot", RdeepBot(num_samples=4, depth=8, rand = random.Random(i*2 + 12)): "Rdeep", Bully(): "Bully", Tibor(): "TibBot", Defense(): "Defense"}

    for opponent in opponents:

        # All bots vs Selfplay
        players = [selfplay, opponent]
        random.shuffle(players)
        winner, points, score = engine.play_game(players[0], players[1], random.Random((i+2)**2))
        #print(f"SelfPlay V {opponent} ==========> {winner}")
        if winner == selfplay:
            if f"SelfPlay V {opponents[opponent]}" not in result_dict:
                result_dict[f"SelfPlay V {opponents[opponent]}"] = 1
            else:
                result_dict[f"SelfPlay V {opponents[opponent]}"] += 1
        
        # All bots vs rand
        if not isinstance(opponent, RandBot):

            players = [opponent, RandBot(random.Random(i))]
            random.shuffle(players)
            winner, points, score = engine.play_game(players[0], players[1], random.Random((i+2)**2))
            #print(f"rand V {opponent} ==========> {winner}")
            if isinstance(winner, RandBot):
                continue

            if f"{opponents[opponent]} V RandBot" not in result_dict:
                result_dict[f"{opponents[opponent]} V RandBot"] = 1

            else:
                result_dict[f"{opponents[opponent]} V RandBot"] += 1
    
        
    print(i)

with open('results.txt', 'w') as convert_file:
    json.dump(result_dict, convert_file, indent=2)




