import os
import numpy as np
from rl_code.Game import Game

def start_rl(mechanic_list):
    agent = init_agent()

    # 1. Load Environment and Q-table structure
    game_env = init_game(mechanic_list)
    Q = np.zeros([game_env.observation_space.n, game_env.action_space.n])
    # game_env.obeservation.n, env.action_space.n gives number of states and action in env loaded
    # 2. Parameters of Q-leanring
    eta = .628
    gma = .9
    epis = 5000
    rev_list = []  # rewards per episode calculate
    # 3. Q-learning Algorithm
    for i in range(epis):
        # Reset environment
        s = game_env.reset()
        rAll = 0
        d = False
        j = 0
        # The Q-Table learning algorithm
        while j < 99:
            game_env.render()

            entity_list = agent.generate_entities()
            entities_with_ids, possible_actions = agent.entity_combinations()
            game_obj = agent.rule_generation()


            j += 1
            # Choose action from Q table
            a = np.argmax(Q[s, :] + np.random.randn(1, game_env.action_space.n) * (1. / (i + 1)))
            # Get new state & reward from environment
            s1, r, d, _ = game_env.step(a) # step forward and simulate current game state
            # Update Q-Table with new knowledge
            Q[s, a] = Q[s, a] + eta * (r + gma * np.max(Q[s1, :]) - Q[s, a])
            rAll += r
            s = s1
            if d == True:
                break
        rev_list.append(rAll)
        game_env.render()

    return game_env


def init_agent():
    return "RL Agent Object"

def init_game(mechanic_list):
    return Game(mechanic_list)