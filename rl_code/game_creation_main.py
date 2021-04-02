import os
import torch
import numpy as np
from pytorch_lightning import Trainer

from trainer import DQNLightning
from rl_code.Game import Game
from visualize import plot_child_entities
from rl_code.simulate import simulate_game
from rl_code.value import get_value_basic
from mechanics.Cards import Deck_of_Cards_Class
from rl_code.Agent import Agent, RandomAgent, CreatorAgent
import time
import gzip, json

from mechanics.Square_Grid_Movement import Square_Grid_Movement_Class


# The different models
from model.entity_combination import do_some_attention, Attention_Model
from model.entity_duplication import Duplicate_Entities_Model
from model.entity_initialization import initialize_some_entities, Initializer_Model
from model.entity_generation import EntityGenerationModel, ManyToOneEncoder

mechanic_types = {
    "Square-Grid Movement": 1,
    "Betting": 2,
}

# We have the entity_creation_net
# the entity_combination_net
# the entity_duplication_net
# the entity_intitializer_net

def save_episode(value,
                 game_env,
                 data_path='/home/jamison/projects/game_generation/data',
                 metadata_path='/home/jamison/projects/game_generation/meta_data',
                 perc_of_data_to_save=.25):

    generator_data = game_env.generator_data
    game_id = '_'.join(time.ctime().split(' ') + [str(time.time())[-6:]])

    # Get the metadata file path
    meta_data_file = os.path.join(metadata_path, 'meta_data_{}.json.gz'.format(game_id))
    with gzip.open(meta_data_file, 'wb') as f:
        len_data = len(generator_data)
        for i, (state, next_state, action) in enumerate(generator_data):
            if np.random.random() <= perc_of_data_to_save:
                state_path = os.path.join(data_path, 'state_{0}_{1}.pt'.format(i, game_id))
                next_state_path = os.path.join(data_path, 'next_state_{0}_{1}.pt'.format(i, game_id))
                L, W = state.shape
                # Set up metadata
                if i == len_data - 1:
                    # If we are at the last decision state
                    reward = value
                    done = 1
                else:
                    # Before the last decision state
                    reward = 0.0
                    done = 0


                my_dict = {"state_path": state_path, "next_state_path": next_state_path, "length": L, "action": int(action), "reward": reward, "value": value, "done": done}
                encoded = (json.dumps(my_dict) + '\n').encode('utf-8')
                f.write(encoded)

                # Save the files
                torch.save(state, state_path)
                torch.save(next_state, next_state_path)


    # Reset the generator data
    game_env.generator_data = []


def collect_experience(args, mechanic_list, creation_runs_until_update=100):
    p = {}  # params
    # p['mechanics'] = mechanic_list
    mechanic_list = ["Square-Grid Movement", "Deck-of-Cards"]  # "Betting"]
    p['mechanics'] = mechanic_list
    # game = Game(mechanic_list)
    # agent = CreatorAgent()

    # attention_model = Attention_Model()

    # 1. Load Environment and Q-table structure
    game_env = init_game(mechanic_list)
    # Q = np.zeros([game_env.observation_space.n, game_env.action_space.n])

    # game_env.obeservation.n, env.action_space.n gives number of states and action in env loaded
    # 2. Parameters of Q-leanring
    values = []  # rewards per episode calculate
    # 3. Q-learning Algorithm
    s = game_env.reset()

    # Generate intitial entities
    # state, trajectories = game.generate_entity_states(agent)
    mechanic_dicts, mechanic_objs = {}, {}
    if "Square-Grid Movement" in mechanic_list:
        Square_Class = Square_Grid_Movement_Class()
        mechanic_dicts[1] = Square_Class.get_mechanic_dict()  # "Square-Grid Movement"
        mechanic_objs[1] = Square_Class  # "Square-Grid Movement"
    if "Deck-of-Cards" in mechanic_list:
        Deck_Class = Deck_of_Cards_Class()
        mechanic_dicts[2] = Deck_Class.get_mechanic_dict()  # "Deck-of-Cards"
        mechanic_objs[2] = Deck_Class  # "Deck-of-Cards"

    # Save the initial model
    model = DQNLightning(**vars(args))
    trainer = Trainer(default_root_dir=args.root_dir)
    trainer.accelerator.setup(trainer, model)
    trainer.save_checkpoint(os.path.join(args.root_dir,"checkpoints","model.ckpt"))

    games_to_reach_min_epsilon = 0
    current_game_num = 0
    while True:
        # Load the model
        print("Loading latest model...")
        model = DQNLightning.load_from_checkpoint(checkpoint_path=os.path.join(args.root_dir,"checkpoints","model.ckpt"))
        model.eval()

        entity_generation_model = model.generation_net
        attention_model = model.combination_net
        duplicate_model = model.duplication_net
        initializer_model = model.initialization_net

        if current_game_num < 0:
            game_env.epsilon = 1.0
        elif current_game_num < games_to_reach_min_epsilon:
            game_env.epsilon = args.eps_start * (games_to_reach_min_epsilon - current_game_num) / games_to_reach_min_epsilon \
                                + args.eps_end * current_game_num / games_to_reach_min_epsilon
        else:
            game_env.epsilon = args.eps_end


        print("Epsilon:", game_env.epsilon)

        with torch.no_grad():
            for i in range(args.creation_runs_until_update):
                # Reset environment

                print("Entity Generation")
                # entity_generation_model = ManyToOneEncoder()
                child_embeddings, child_trajectories, parent_embeddings, parent_trajectories = game_env.generate_entity_states(
                    entity_generation_model, mechanic_dicts)
                all_embeddings = torch.cat((parent_embeddings, child_embeddings), dim=0)
                all_trajectories = parent_trajectories + child_trajectories

                # ---------------------------------- ATTENTION FOR ENTITY COMBINATION --------------------------------------
                print("Entity Combination")
                # attention_model = Attention_Model()
                indices_to_combine, new_embeddings, comb_to_emb_map = do_some_attention(all_embeddings, all_trajectories,
                                                                                        attention_model)
                # indices_to_combine_child, child_embeddings, child_comb_to_emb_map = do_some_attention(child_embeddings, child_trajectories, attention_model, is_child=True)
                # indices_to_combine_parent, parent_embeddings, parent_comb_to_emb_map = do_some_attention(parent_embeddings, parent_trajectories, attention_model, is_child=False)

                # ---------------------------------------- ENTITY DUPLICATION ----------------------------------------------
                print("Entity Duplication")
                # duplicate_model = Duplicate_Entities_Model(mechanic_types, mechanic_dicts)
                duplicate_combined_dict = duplicate_model.transformer_duplicate(new_embeddings, all_trajectories,
                                                                                comb_to_emb_map)
                duplicated_embeddings = duplicate_embeddings(new_embeddings, duplicate_combined_dict)


                # ----------------------------------------- ENTITY CREATION ------------------------------------------------
                entity_obj_dict = game_env.create_entity_objects(duplicate_combined_dict, all_trajectories, mechanic_objs,
                                                                 new_embeddings)


                # ------------------------------------- ENTITY PLACE INITIALIZATION ----------------------------------------
                print("Entity Initialization")
                # initializer_model = Initializer_Model()
                entity_obj_dict = initialize_some_entities(entity_obj_dict, initializer_model, duplicated_embeddings,
                                                           duplicate_combined_dict)

                # ----------------------------------------- ENTITY GROUP CREATION ------------------------------------------
                entity_groups, parents_to_groups, actions_to_parents = game_env.create_entity_groups(entity_obj_dict,
                                                                                                     mechanic_objs,
                                                                                                     mechanic_types)

                # -------------------------------------- INITIALIZE GAME ---------------------------------------------------
                game_obj = game_env.create_game_obj(entity_obj_dict, entity_groups, mechanic_objs, mechanic_types,
                                                    parents_to_groups, actions_to_parents)

                # ------------------------------------- GET VALUE ----------------------------------------------------------
                print("Getting value")
                value = get_value_basic(game_obj)
                print(value)
                save_episode(value, game_env)
                current_game_num += 1
                print()


def duplicate_embeddings(embeddings, duplicate_combined_dict):
    duped_embeddings = []
    for i, key in enumerate(duplicate_combined_dict):
        dup_num = duplicate_combined_dict[key][0]  # gets the duplication number for this embedding
        embedding = embeddings[i]
        for d in range(dup_num):
            duped_embeddings.append(embedding.clone())
    return torch.stack(duped_embeddings)


def init_agent(p):
    return CreatorAgent()


def init_game(mechanic_list):
    return Game(mechanic_list)
