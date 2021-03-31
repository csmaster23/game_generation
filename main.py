import os
import sys

from rl_code.game_creation_main import collect_experience
from rl_code.rl_main import start_rl
from game_embedding.random_mechanics import generate_random_mechanics



def main(load_game, sim_type):
    if load_game is None:
        num_mechanics = 2
        game_mechanics = generate_random_mechanics(num_mechanics)

        # final_game_obj = start_rl(game_mechanics)

        collect_experience(game_mechanics)

        # save finalized game object

    else:
        if sim_type == 'human':
            pass # simulate game with human player
        else:
            pass # computer simulate game and output results

    return





if __name__ == "__main__":
    # execute only if run as a script
    main(None, 'human')