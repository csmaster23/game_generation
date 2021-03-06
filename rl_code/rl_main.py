import os
import torch
import numpy as np
from rl_code.Game import Game
from visualize import plot_child_entities
from rl_code.simulate import simulate_game
from rl_code.value import get_value_basic
from mechanics.Cards import Deck_of_Cards_Class
from rl_code.Agent import Agent, RandomAgent, CreatorAgent
from rl_code.Duplicate_Entities import Duplicate_Entities_Model
from rl_code.Attention import do_some_attention, Attention_Model
from mechanics.Square_Grid_Movement import Square_Grid_Movement_Class
from rl_code.Initialize_Entities import initialize_some_entities, Initializer_Model

mechanic_types = {
  "Square-Grid Movement" : 1,
  "Betting"       : 2,
}

def start_rl(mechanic_list):
    p = {}  # params
    # p['mechanics'] = mechanic_list
    mechanic_list = ["Square-Grid Movement", "Deck-of-Cards"] #"Betting"]
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
            mechanic_dicts, mechanic_objs = {}, {}
            if "Square-Grid Movement" in mechanic_list:
                Square_Class = Square_Grid_Movement_Class()
                mechanic_dicts[1] = Square_Class.get_mechanic_dict() # "Square-Grid Movement"
                mechanic_objs[1] = Square_Class # "Square-Grid Movement"
            if "Deck-of-Cards" in mechanic_list:
                Deck_Class = Deck_of_Cards_Class()
                mechanic_dicts[2] = Deck_Class.get_mechanic_dict() # "Deck-of-Cards"
                mechanic_objs[2] = Deck_Class # "Deck-of-Cards"
            child_embeddings, child_trajectories, parent_embeddings, parent_trajectories = game_env.generate_entity_states(agent, mechanic_dicts)
            all_embeddings = torch.cat((parent_embeddings, child_embeddings), dim=0)
            all_trajectories = parent_trajectories + child_trajectories

            # ---------------------------------- ATTENTION FOR ENTITY COMBINATION --------------------------------------
            attention_model = Attention_Model()
            indices_to_combine, new_embeddings, comb_to_emb_map = do_some_attention(all_embeddings, all_trajectories, attention_model)
            # indices_to_combine_child, child_embeddings, child_comb_to_emb_map = do_some_attention(child_embeddings, child_trajectories, attention_model, is_child=True)
            # indices_to_combine_parent, parent_embeddings, parent_comb_to_emb_map = do_some_attention(parent_embeddings, parent_trajectories, attention_model, is_child=False)

            # ---------------------------------------- ENTITY DUPLICATION ----------------------------------------------
            duplicate_model = Duplicate_Entities_Model(mechanic_types, mechanic_dicts)
            duplicate_combined_dict = duplicate_model.transformer_duplicate(new_embeddings, all_trajectories, comb_to_emb_map)
            duplicated_embeddings = duplicate_embeddings( new_embeddings, duplicate_combined_dict )
            # parent_duplicate_combined_dict = duplicate_model.transformer_duplicate(parent_embeddings, parent_trajectories, parent_comb_to_emb_map, is_child=False)
            # child_duplicate_combined_dict = duplicate_model.transformer_duplicate( child_embeddings, child_trajectories, child_comb_to_emb_map, is_child=True )

            # ----------------------------------------- ENTITY CREATION ------------------------------------------------
            entity_obj_dict = game_env.create_entity_objects(duplicate_combined_dict, all_trajectories, mechanic_objs, new_embeddings)

            # plot_child_entities(entity_obj_dict)

            # ------------------------------------- ENTITY PLACE INITIALIZATION ----------------------------------------
            initializer_model = Initializer_Model()
            entity_obj_dict = initialize_some_entities(entity_obj_dict, initializer_model, duplicated_embeddings, duplicate_combined_dict)


            # ----------------------------------------- ENTITY GROUP CREATION ------------------------------------------
            entity_groups, parents_to_groups, actions_to_parents = game_env.create_entity_groups(entity_obj_dict, mechanic_objs, mechanic_types)
            # entity_groups['Square-Grid Movement 1'].visualize(entity_obj_dict)


            # -------------------------------------- INITIALIZE GAME ---------------------------------------------------
            game_obj = game_env.create_game_obj(entity_obj_dict, entity_groups, mechanic_objs, mechanic_types, parents_to_groups, actions_to_parents)

            value = get_value_basic(game_obj)



            simulate_game(game_obj)

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

def duplicate_embeddings( embeddings, duplicate_combined_dict ):
    duped_embeddings = []
    for i, key in enumerate(duplicate_combined_dict):
        dup_num = duplicate_combined_dict[key][0] # gets the duplication number for this embedding
        embedding = embeddings[i]
        for d in range(dup_num):
            duped_embeddings.append(embedding.clone())
    return torch.stack(duped_embeddings)


def init_agent(p):
    return CreatorAgent()

def init_game(mechanic_list):
    return Game(mechanic_list)
