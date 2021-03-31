import random

class Random_Agent:
    def __init__(self, id=None, verbose=True):
        self.id = id
        self.player_name = "player_" + str(self.id)
        self.verbose = verbose

    def choose_action(self, action_dict, ):
        if self.verbose:
            print("Action Dict: %s" % str(action_dict))
        indices = list(action_dict.keys())
        if len(indices) == 0:
            return None
        choice = random.choice(indices)
        if self.verbose:
            print("Player %s chooses action indice %s that is action type %s" % (str(self.id), str(choice), str(action_dict[choice]['action_type'])))
        return choice

    def __str__(self):
        str_builder = "Random Agent\nPlayer ID: " + str(self.id)
        return str_builder
