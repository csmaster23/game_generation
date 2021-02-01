import os
import sys
from rl_code.rl_main import start_rl
from game_embedding.random_mechanics import generate_random_mechanics



def main():
    num_mechanics = 2
    game_mechanics = generate_random_mechanics(num_mechanics)

    rl_game_object = start_rl(game_mechanics)












if __name__ == "__main__":
    # execute only if run as a script
    main()