import argparse
import os
import sys

import torch
import numpy as np

from rl_code.game_creation_main import collect_experience
from rl_code.rl_main import start_rl
from game_embedding.random_mechanics import generate_random_mechanics




def main(args, load_game, sim_type):
    if load_game is None:
        num_mechanics = 2
        game_mechanics = generate_random_mechanics(num_mechanics)

        # final_game_obj = start_rl(game_mechanics)

        collect_experience(args, game_mechanics)

        # save finalized game object

    else:
        if sim_type == 'human':
            pass # simulate game with human player
        else:
            pass # computer simulate game and output results

    return





if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--env", type=str, default="CartPole-v0", help="gym environment tag")
    parser.add_argument("--gamma", type=float, default=1, help="discount factor")
    parser.add_argument("--sync_rate", type=int, default=10,
                        help="how many frames do we update the target network")
    parser.add_argument("--replay_size", type=int, default=1000,
                        help="capacity of the replay buffer")
    parser.add_argument("--warm_start_size", type=int, default=1000,
                        help="how many samples do we use to fill our buffer at the start of training")
    parser.add_argument("--eps_last_frame", type=int, default=1000,
                        help="what frame should epsilon stop decaying")
    parser.add_argument("--eps_start", type=float, default=1.0, help="starting value of epsilon")
    parser.add_argument("--eps_end", type=float, default=0.01, help="final value of epsilon")
    parser.add_argument("--episode_length", type=int, default=200, help="max length of an episode")
    parser.add_argument("--max_episode_reward", type=int, default=200,
                        help="max episode reward in the environment")
    parser.add_argument("--warm_start_steps", type=int, default=1000,
                        help="max episode reward in the environment")
    parser.add_argument("--root_dir", type=str, default='/home/jamison/projects/game_generation/results/my_model')
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--creation_runs_until_update", type=int, default=25)

    args = parser.parse_args()

    main(args, None, 'human')
    # execute only if run as a script
    # main(None, 'human')