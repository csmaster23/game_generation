import os
import numpy as np
from rl_code.Game import Game
from rl_code.Agent import Agent, RandomAgent, CreatorAgent
from rl_code.Duplicate_Entities import Duplicate_Entities_Model
from rl_code.Attention import do_some_attention, Attention_Model

mechanic_types = {
  "Square-Grid Movement" : 1,
  "Betting"       : 2,
}

def start_rl(mechanic_list):
    p = {}  # params
    # p['mechanics'] = mechanic_list
    mechanic_list = ["Square-Grid Movement", "Betting"]
    p['mechanics'] = mechanic_list
    # game = Game(mechanic_list)
    # agent = CreatorAgent()
    agent = init_agent(p)
    # attention_model = Attention_Model()

    # 1. Load Environment and Q-table structure
    game_env = init_game(mechanic_list)
    # Q = np.zeros([game_env.observation_space.n, game_env.action_space.n])
    Q = np.zeros([5,5])
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

            # Generate intitial entities
            # state, trajectories = game.generate_entity_states(agent)
            child_embeddings, child_trajectories, parent_embeddings, parent_trajectories = game_env.generate_entity_states(agent)

            # ---------------------------------- ATTENTION FOR ENTITY COMBINATION --------------------------------------
            attention_model = Attention_Model()
            indices_to_combine_child, child_embeddings, child_comb_to_emb_map = do_some_attention(child_embeddings, child_trajectories, attention_model, is_child=True)
            indices_to_combine_parent, parent_embeddings, parent_comb_to_emb_map = do_some_attention(parent_embeddings, parent_trajectories.tolist(), attention_model, is_child=False)


            # ---------------------------------------- ENTITY DUPLICATION ----------------------------------------------
            duplicate_model = Duplicate_Entities_Model(mechanic_types)
            parent_duplicate_combined_dict = duplicate_model.transformer_duplicate(parent_embeddings, parent_trajectories.tolist(), parent_comb_to_emb_map, is_child=False)
            child_duplicate_combined_dict = duplicate_model.transformer_duplicate( child_embeddings, child_trajectories, child_comb_to_emb_map, is_child=True )



            # entity_list = game_env.generate_entities(state, trajectories)
            # Combine entities
            # combo_states = game_env.generate_entity_combo_states(agent, state)
            final_entity_list = game_env.combine_entities(state)

            # Generate game rules
            game_obj = game_env.rule_generation(agent, final_entity_list)

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


def init_agent(p):
    return CreatorAgent()

def init_game(mechanic_list):
    return Game(mechanic_list)
