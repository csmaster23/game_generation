

class Random_Agent:
    def __init__(self, id=None):
        self.id = id
        self.player_name = "player_" + str(self.id+1)



    def __str__(self):
        str_builder = "Random Agent\nPlayer ID: " + str(self.id)
        return str_builder
