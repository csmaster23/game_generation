import gym
import numpy as np
# from gym import spaces
from rl_code.Agent import RandomAgent, CreatorAgent
from torch import nn
import torch
from mechanics.Entity import Entity
from mechanics.Entity_Group import EntityGroup

mechanic_types = {
  "Square-Grid Movement" : 1,
  "Deck-of-Cards"       : 2,
}
num_mechanic_types = {
  1: "Square-Grid Movement",
  2: "Deck-of-Cards",
}

interpret_level = {
      0: "mechanic_num",
      1: "num_groups",
      2: "selected_group",
      3: "selected_parent_entity", # Defaults to 0
      4: "num_child_entities",
      5: "selected_child_entity",
      6: "num_action_types",
      7: "selected_action_type",
      8: "num_patterns",
      9: "selected_pattern",
      10: "pattern_length",
      11: "pattern_symbol"
    }


class GameObject:
  def __init__(self, entity_object_dict, entity_groups, parents_to_groups, actions_to_parents, objective=None, num_players=2):
    """
    Creates a game object that an agent can interact with. Defines the legal actions and movements for each piece type.
    :param entity_object_dict:
    :param entity_groups:
    :param parents_to_groups:
    :param actions_to_parents:
    :param objective:
    :param num_players:
    """
    self.entity_object_dict = entity_object_dict
    self.entity_groups = entity_groups
    self.tracker_dict = self.generate_trackers(entity_object_dict, entity_groups)
    print(self.tracker_dict)
    self.parents_to_groups = parents_to_groups
    self.actions_to_parents = actions_to_parents

    # Take out specific entities from the gameplay
    self.available_entity_dict = dict()
    self.available_parents = set()
    self.available_actions = dict()
    self.available_children_ids = set()
    self.available_entity_group_names = set()
    for key in entity_object_dict:
      entity = entity_object_dict[key]
      use = False
      for name in entity.entity_names:
        if "square" in name or "square_movement" in name or "reserve" in name:
          use = True
          if ("square" in name or "reserve" in name) and "square_movement" not in name:
            self.available_parents.add(name)
          else:
            self.available_children_ids.add(key)
      if use:
        self.available_entity_dict[key] = entity_object_dict[key]
        # Add the possible actions to a set
        self.available_entity_dict[key].possible_actions = set(self.available_entity_dict[key].actions_to_patterns.keys())

    self.available_entity_ids = list(self.available_entity_dict.keys())
    self.available_parents = list(self.available_parents)
    self.available_children_ids = list(self.available_children_ids)

    # Now set up the available actions per parent
    for parent_name in self.available_parents:
      entity_group = self.entity_groups[self.parents_to_groups[parent_name]]
      self.available_actions[parent_name] = list(entity_group.adj_matrices[parent_name].keys())

    # Gets the entity groups we want. Here we only care about square grid movement
    self.available_entity_group_names = {name for name in self.entity_groups if "Square-Grid Movement" in name}
    self.available_entity_groups = dict()
    for group_name in self.available_entity_group_names:
      entity_group = self.entity_groups[group_name]
      self.available_entity_groups[group_name] = entity_group

    # Capture piece_x, Capture all currently the only options
    if objective is None:
      self.objective = {"type": "Capture", "target": "all"}
    else:
      self.objective = objective
    self.num_players = num_players
    self.player_ids = {"player_{}".format(i+1): i for i in range(num_players)}

    # Add complicated patterns to adjacency matrices based on the pieces we are using
    self.create_pattern_adj_matrices()

    # Set starting game state
    self.game_state = {"turn": "player_1"}
    self.sq_reserve_id = None

    # Check to make sure the game over function works
    results = self.check_game_over()
    # Check to make sure we can generate the action vector
    action_vector, index_to_action = self.get_all_legal_actions(self.game_state)
    # action_key = list(index_to_action.keys())[0]
    # self.move(index_to_action[action_key]['target_id'], index_to_action[action_key]['destination_id'])
    print()

  def get_game_state(self):
    return self.game_state
  def set_game_state(self, state):
    self.game_state = state

  def create_pattern_adj_matrices(self):
    for key in self.available_entity_ids:
      entity = self.available_entity_dict[key]
      for action_type in entity.actions_to_patterns:
        patterns = entity.actions_to_patterns[action_type]
        for pattern_key in patterns:
          pattern = patterns[pattern_key]
          if len(pattern) > 1:
            tuple_pattern = tuple(pattern)
            parent = self.actions_to_parents[action_type]
            group_name = self.parents_to_groups[parent]
            if tuple_pattern not in self.entity_groups[group_name].adj_matrices[parent][action_type]:
              cur_adj_matrix = self.entity_groups[group_name].adj_matrices[parent][action_type][pattern[0]]
              for symbol in pattern[1:]:
                cur_adj_matrix = cur_adj_matrix @ self.entity_groups[group_name].adj_matrices[parent][action_type][symbol]
              self.entity_groups[group_name].adj_matrices[parent][action_type][tuple_pattern] = cur_adj_matrix

  def get_reserve_id(self):
    if self.sq_reserve_id is None:
      for key in self.entity_object_dict:
        if 'reserve_1' in self.entity_object_dict[key].entity_names:
          return key                                                              # returns the id of the reserve square
    else:
      return self.sq_reserve_id

  def find_opponent_id_by_parent(self, parent_id):
    parent = self.entity_object_dict[parent_id]
    for id in parent.my_stored_ids:
      for name in self.entity_object_dict[id].entity_names:
        if 'square' in name:
          return id
    return None

  def move(self, target_id, destination_id):
    # Get the old location of the target
    old_location = self.available_entity_dict[target_id].storage_location
    # Remove entity id from old location
    self.available_entity_dict[old_location].my_stored_ids.remove(target_id)
    # Move the entity to that location
    self.available_entity_dict[target_id].storage_location = destination_id
    # Put entity id in the new location
    self.available_entity_dict[destination_id].my_stored_ids.append(target_id)
    # Update tracker dict
    self.tracker_dict = self.generate_trackers(self.entity_object_dict, self.entity_groups)


  def execute_action(self, chosen_action): # chosen action comes in as dict
    # print("Chosen action: %s" % str(chosen_action))
    if 'capture' in chosen_action['action_type']: # capture move
      reserve_id = self.get_reserve_id()
      # move the other piece to reserve
      opponent_piece_id = self.find_opponent_id_by_parent(chosen_action['destination_ids'][-1])
      assert opponent_piece_id is not None
      self.move(opponent_piece_id, reserve_id) # moves captured piece to reserve

    self.move(chosen_action['target_id'], chosen_action['destination_ids'][-1]) # moves current player piece to captured spot
    return

  def check_if_capture_possible(self, action_type, target, dest):
    # TODO: Finish this method
    return True

  def check_move_action_validity(self, action_type, target, dest):
    # TODO: Finish this method
    return True

  def check_if_valid(self, action_dict):
    # print("Action dict: %s" % str(action_dict))
    action_type = action_dict['action_type']
    if 'remove_captured_piece' in action_type:      # we do not allow agents to take this action
      return False
    elif 'capture' in action_type:                  # is this specific capture action possible
      return self.check_if_capture_possible(action_type, action_dict['target_id'], action_dict['destination_ids'][-1])
    else:                                           # is this specific movement action possible
      return self.check_move_action_validity(action_type, action_dict['target_id'], action_dict['destination_ids'][-1])

  def get_actions_for_player(self, player_name):
    actions, indices = self.get_all_legal_actions(self.game_state)
    valid_indices = {}

    for i, action in enumerate(indices):
      is_valid = self.check_if_valid(indices[action])
      if is_valid:
        valid_indices[action] = indices[action]

    return valid_indices


  def get_all_legal_actions(self, game_state):
    """
    This function goes through and gets all the legal actions and puts it into a vector along with a dictionary that goes from the possible move
    index to something that can be easily fed into the game
    :param game_state:
    :return: action_vector (list), index_to_action (dictionary that maps each index in the action vector to an interpretable action)
    """
    player_id = game_state["turn"]
    index = 0
    action_vector = []
    index_to_action = dict()
    for key in self.available_children_ids:
      entity = self.available_entity_dict[key]
      # First get the parent
      for parent_name in self.available_parents:
        # Then get the possible action types for that parent
        for action_type in self.available_actions[parent_name]:
          # If our piece can do that action
          entity_group = self.available_entity_groups[self.parents_to_groups[parent_name]]
          # Get the target_entity_index
          possible_moves_set = set()
          all_possible_moves = []
          if action_type in entity.possible_actions and player_id in entity.entity_names:
            # Get where the current entity is stored
            storage_location_id = entity.storage_location
            # Get the index of that parent from the entity group
            parent_idx = entity_group.id_to_idx[storage_location_id]
            # Get the patterns from the current action type
            patterns = entity.actions_to_patterns[action_type]
            destination_ids_dict, idx_to_pattern = dict(), dict()
            # Cycle through the patterns
            for pattern_key in patterns:
              pattern = patterns[pattern_key]
              # if len(pattern) > 1:
              #   pattern = tuple(pattern)
              # else:
              #   pattern = pattern[0]
              # Get a vector that determines the movements we can make
              destination_ids = []
              current_pattern = []
              new_parent_idx = parent_idx
              for pos, symbol in enumerate(pattern):
                movement_vector = entity_group.adj_matrices[parent_name][action_type][symbol][new_parent_idx]
                possible_moves = np.where(movement_vector==1)[0]
                if len(possible_moves) > 1:
                  # This means the pattern is only 1 long
                  all_possible_moves += list(possible_moves)
                  for move in possible_moves:
                    destination_id = entity_group.idx_to_id[move]
                    destination_ids_dict[move] = [destination_id]
                    idx_to_pattern[move] = [symbol]
                elif len(possible_moves) == 1:
                  # This means the pattern could be more than one long
                  destination_id = entity_group.idx_to_id[possible_moves[0]]
                  destination_ids.append(destination_id)
                  current_pattern.append(symbol)
                  len_pattern = len(pattern)
                  if "drag" in action_type or pos == len_pattern - 1:
                    all_possible_moves += list(possible_moves)
                    # Keeps track of each possible place we could stop at
                    destination_ids_dict[possible_moves[0]] = destination_ids[:]
                    idx_to_pattern[possible_moves[0]] = current_pattern[:]
                  new_parent_idx = possible_moves[0]
                else:
                  break

            possible_moves_set = set(all_possible_moves)

          for target_entity in range(entity_group.num_entities):
            if target_entity in possible_moves_set:
              action_vector.append(1)
              index_to_action[index] = {"target_id": key, "destination_ids": destination_ids_dict[target_entity], "action_type": action_type, "pattern" : idx_to_pattern[target_entity]}
            else:
              action_vector.append(0)
            index += 1

    return action_vector, index_to_action

  def check_list(self, l):
    num_trues = 0
    for item in l:
      if item:
        num_trues += 1

    if num_trues > 1:
      return False
    else:
      return True

  def check_game_over(self):
    # Eventually move this function to the mechanic portion
    type = self.objective['type']
    target = self.objective['target']
    players_in_game = [False for _ in range(self.num_players)]
    game_over = True

    if type == "Capture":
      # Loop through to see if the game is over
      for key in self.tracker_dict:
        if 'square' in key:
          for square_idx in self.tracker_dict[key]:
            for piece_names in self.tracker_dict[key][square_idx]:
              if target in piece_names or target == "all":
                for player_name in self.player_ids:
                  if player_name in piece_names:
                    players_in_game[self.player_ids[player_name]] = True
                    game_over = self.check_list(players_in_game)
                    break
              if not game_over:
                break
            if not game_over:
              break
        if not game_over:
          break

    # Gets the players that are still in the game
    players_still_in_game = {player_id : players_in_game[self.player_ids[player_id]] for player_id in self.player_ids}
    return game_over, players_still_in_game





  def generate_trackers(self, entity_object_dict, entity_groups):
    tracker_dict = dict()

    # Create a human readable tracker format for the game so that we can easily check that game mechanics are working
    # Also use this to check winning conditions
    for key in entity_groups:
      group = entity_groups[key]
      parent_ids = group.parents_to_ids
      for parent_type in parent_ids:
        tracker_dict[parent_type] = dict()
        for parent_id in parent_ids[parent_type]:
          parent_idx = entity_groups[key].id_to_idx[parent_id]

          # Store the names of the children instead of their ids
          child_info = []

          try:
            for id in entity_object_dict[parent_id].my_stored_ids:
              child_info.append(entity_object_dict[id].entity_names)
          except KeyError:
            print()

          tracker_dict[parent_type][parent_idx] = child_info

    return tracker_dict




class Game():#gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, mechanic_list, verbose=True):
    super(Game, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    N_DISCRETE_ACTIONS = 0
    self.num_possible_mechanics = 2
    self.max_level = 12
    # self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
    # Example for using image as input:
    HEIGHT = 0
    WIDTH = 0
    N_CHANNELS = 0
    self.max_options = 10
    # self.observation_space = spaces.Box(low=0, high=255, shape=
    #                 (HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
    self.mechanic_list = mechanic_list
    self.verbose=verbose
    self.game_obj = None

  def step(self, action):
      # Execute one time step within the environment
      r = self.game_obj.simulate()
      s1 = self.represent_state()
      d = False
      return s1, r, d, 0.0
  def reset(self):
    # Reset the state of the environment to an initial state
    return 'resetted'
  def render(self, mode='human', close=False):
    # Render the environment to the screen
    return 'rendered'
  def simulate(self):
    return 0.0
  def represent_state(self):
    return self


  def create_game_obj(self, entity_obj_dict, entity_groups, mechanic_objs, mechanic_types, parents_to_groups, actions_to_parents):
    """
    Creates the entity game object
    :param entity_obj_dict:
    :param mechanic_objs:
    :param mechanic_types:
    :return:
    """

    game_obj = GameObject(entity_obj_dict, entity_groups, parents_to_groups, actions_to_parents)
    return game_obj



  def create_entity_objects(self, duplicate_dict, trajectories, mechanic_objs, embeddings):
    print("Top of create entities")
    # TODO: Need to add emebedding values to the entity objects
    # Adds the trajectory info the entities that originals
    for key in duplicate_dict.keys():
      if len(duplicate_dict[key][1]) == 0:
        duplicate_dict[key][1].append(key)

    all_entities = {}
    for element in duplicate_dict.keys():
      embedding = embeddings[element] # grabs the corresponding embedding for this given entity
      for d in range(duplicate_dict[element][0]):  # number of times to duplicate
        entity = Entity()
        entity.set_embedding(embedding.clone())
        for needed_traj in duplicate_dict[element][1]: # references entitiy needed trajectory list
          # needed traj is a number that can be indexed
          traj = trajectories[needed_traj]
            # this gets the class object
          if len(traj.shape) == 1:
            traj = traj.unsqueeze(0)
          for t in traj:
            mechanic_num = t[0].item()
            obj = mechanic_objs[mechanic_num]
            entity = obj.add_to_entity(entity, t)
        all_entities[entity.id] = entity

      else:                                     # means that are referencing combined entities
        pass

    return all_entities

  def create_entity_groups(self, entity_object_dict, mechanic_objs, mechanic_types):
    entity_group_dict = dict()
    for entity_id in entity_object_dict:
      entity = entity_object_dict[entity_id]
      for group in entity.entity_groups:
        # Create the entity groups
        try:
          entity_group_obj = entity_group_dict[group[1]]
        except KeyError:
          entity_group_obj = EntityGroup(group[1])
          entity_group_dict[group[1]] = entity_group_obj
        entity_group_obj.add_entity_to_group(entity)

    for group_name in entity_group_dict:
      for mechanic_type in mechanic_types:
        if mechanic_type in group_name:
          mechanic_num = mechanic_types[mechanic_type]
          mechanic_obj = mechanic_objs[mechanic_num]
          break
      entity_group_dict[group_name].assign_entity_indices(entity_object_dict)
      entity_group_dict[group_name].create_adjacency_matrices(mechanic_obj)

    parent_to_entity_group = dict()
    actions_to_parents = dict()
    for group_name in entity_group_dict:
      for parent_name in entity_group_dict[group_name].parents_to_ids:
        parent_to_entity_group[parent_name] = group_name
        actions = entity_group_dict[group_name].adj_matrices[parent_name]
        for action in actions:
          actions_to_parents[action] = parent_name

      # Record how many entities are in the group
      entity_group = entity_group_dict[group_name]
      entity_group.num_entities = len(entity_group.idx_to_id)

    return entity_group_dict, parent_to_entity_group, actions_to_parents


























  def generate_entities(self, entity_states):
    print("\n--- Top of Generate Entities ---")
    entity_list = []
    max_dict = { 0:2, 1:1, 2:1, 3:3, 4:2, 5:3, 6:9 }
    #            mec  grid sq   typ  pat  len  sym
    entity_states = np.asarray( [
      [ 1,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,      0,1,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0 ], # mechanic one-hot encoding
      [ 1,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,      1,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0 ], # num grids
      [ 1,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,      1,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0 ], # square types
      [ 0,0,1,0,0,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,1,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,      0,1,0,0,0,0,0,0,0, 0,0,0,0,0,1,0,0,0, 0,0,0,1,0,0,0,0,0, 0,0,0,0,0,0,0,0,0 ], # piece types max 6
      [ 0,1,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 0,1,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,      0,1,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0 ], # num patterns
      [ 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,      0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0 ], # pattern len
      [ 0,1,0,0,0,0,0,0,0, 0,0,0,0,0,1,0,0,0, 0,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,0,      0,0,0,0,0,0,0,1,0, 0,0,0,1,0,0,0,0,0, 0,0,0,0,0,0,0,1,0, 0,0,0,0,1,0,0,0,0 ]  # symbol
    ] )
    print("Entity States: %s" % str(entity_states))
    mec_state_list = lst = np.hsplit(entity_states, len(self.mechanic_list)) # this splits the giant array into mechanic parts
    for mec in mec_state_list:
      entity_list.append( self.entity_list_per_mechanic(mec, max_dict) )

    return entity_list

  def entity_list_per_mechanic(self, state, max_dict):
    print("Top of Entity List for each mechanic")
    print("Max dict: %s" % str(max_dict))
    for row in state:
      print(row)
    entities = []
    max_value = max_dict[max(max_dict, key=max_dict.get)]
    print("Max valuez: %s" % str(max_value))

    return entities


  def generate_entity_states(self, agent, mechanic_dicts):
    """
    :param agent:
    :return:
    entity_embeddings (N, dim): A tensor of entity embeddings
    grouped_tree_trajectories (N, ?, max_level+1): A list of tensors with tree trajectories that describe how to create the different entities
    """
    # two_mechanic_dicts = self.get_mechanic_dicts()

    # We have a list of embeddings that represent the leaves of the tree. This is the current state
    # Are converted to embeddings before they are fed into the transformer
    tree_trajectories = [torch.LongTensor([[0]*self.max_level])]

    # Make sure we have enough slots to encode the mechanic types
    entity_markers = []
    # Run the RL Agent
    for i, mechanic in enumerate(self.mechanic_list):
      tree_trajectories[-1][0, 0] = mechanic_types[mechanic]
      cur_dict = mechanic_dicts[mechanic_types[mechanic]]

      # Recurse down until we have made all the selections we need
      self.make_agent_selections(agent, tree_trajectories, cur_dict, 0, entity_markers)

    tree_trajectories.pop() # The last tree trajectory is extra
    tree_trajectories = torch.stack(tree_trajectories)

    if self.verbose:
      print(tree_trajectories)

    # Extract entity embeddings
    child_embeddings, child_trajectories, parent_embeddings, parent_trajectories = self.get_entity_embeddings(agent, tree_trajectories, mechanic_dicts)

    return child_embeddings, child_trajectories, parent_embeddings, parent_trajectories

  def get_grouped_tree_trajectories(self, tree_trajectories, entity_markers):
    grouped_tree_trajectories = []
    for i, idx in enumerate(entity_markers):
      if i == len(entity_markers) - 1:
        stop = -1
      else:
        stop = entity_markers[i + 1]
      # grouped_tree_trajectories.append(tree_trajectories[entity_markers[i]:stop].reshape(-1,self.max_level+1))
    return grouped_tree_trajectories

  def get_entity_embeddings(self, agent, tree_trajectories, mechanic_dicts):
    entity_embeddings = []
    N = tree_trajectories.shape[0]
    tree_trajectories = tree_trajectories.reshape(N, -1)

    # Get the different items we need to determine entity types
    mechanic_nums = tree_trajectories[:,0]
    group_nums = tree_trajectories[:,2]
    entity_nums = tree_trajectories[:,5]

    # Get the indices for the different entities
    child_entity_indices = np.where((entity_nums[:-1] != entity_nums[1:]) | (group_nums[:-1] != group_nums[1:]) | (mechanic_nums[:-1] != mechanic_nums[1:]))[0]
    child_entity_indices += 1
    parent_entity_indices = np.where((group_nums[:-1] != group_nums[1:]) | (mechanic_nums[:-1] != mechanic_nums[1:]))[0]
    parent_entity_indices += 1

    # Get parent embeddings and trajectories
    parent_entity_indices = np.concatenate([np.array([0]), parent_entity_indices])
    parent_trajectories = []
    for idx in parent_entity_indices:
      mechanic_type = tree_trajectories[idx, 0].item()
      for num_parent in range(1, mechanic_dicts[mechanic_type]["num_parent_entity_types"]+1):
        parent_embedding = tree_trajectories[idx].clone()
        parent_embedding[3] = num_parent
        parent_embedding[4:] = 0
        parent_trajectories.append(parent_embedding)
    parent_trajectories = torch.stack(parent_trajectories)

    with torch.no_grad():
      N = tree_trajectories.shape[0]
      tree_embeddings = agent.gen_embedder(tree_trajectories).reshape(N,-1)
      N = parent_trajectories.shape[0]
      parent_embeddings = agent.gen_embedder(parent_trajectories).reshape(N,-1)

    # Now get the child entities and trajectories
    start_idx = 0
    child_entity_embeddings = []
    child_trajectories = []
    for end_idx in child_entity_indices:
      child_trajectories.append(tree_trajectories[start_idx:end_idx,:])
      child_entity_embeddings.append(torch.mean(tree_embeddings[start_idx:end_idx,:],dim=0))
      start_idx = end_idx
    child_entity_embeddings.append(torch.mean(tree_embeddings[start_idx:, :], dim=0))
    child_trajectories.append(tree_trajectories[start_idx:, :])
    child_embeddings = torch.stack(child_entity_embeddings)

    return child_embeddings, child_trajectories, parent_embeddings, [parent_trajectory for parent_trajectory in parent_trajectories]

  def make_agent_selections(self, agent, tree_trajectories, cur_dict, level, iteration=None):

    new_level = level + 1

    # Need to zero out all entries so transformer knows where we are on the tree
    if tree_trajectories[-1][0, new_level] != 0:
      tree_trajectories[-1][0, new_level:] = 0

    if interpret_level[new_level] in ["selected_group", "selected_child_entity", "selected_pattern"]: # These actions are not selected
      # Update our state representation
      tree_trajectories[-1][0, new_level] = iteration
      if self.verbose:
        print("On level:", new_level, "it is iteration", iteration)
      self.make_agent_selections(agent, tree_trajectories, cur_dict, new_level, iteration)
    elif interpret_level[new_level] in ["selected_parent_entity"]:
      self.make_agent_selections(agent, tree_trajectories, cur_dict, new_level, iteration)
    elif interpret_level[new_level] in ["selected_action_type"]: # These actions need to be selected and masked out
      probs = agent.take_action(tree_trajectories, self.selected_action_mask)
      selection = torch.argmax(probs).item()
      self.selected_action_mask[selection] = float("-inf")
      if self.verbose:
        print("On action selection. The agent chose", selection)

      # Update our state representation
      tree_trajectories[-1][0, new_level] = selection
      self.make_agent_selections(agent, tree_trajectories, cur_dict, new_level, iteration)


    else:
      # Get the min and max options
      cur_min, cur_max = cur_dict[interpret_level[new_level]]

      # Create a mask for the agent
      options = np.arange(self.max_options)
      mask = torch.zeros(self.max_options)
      mask[np.where((options < cur_min) | (options > cur_max))[0]] = float("-inf")

      if interpret_level[new_level] == "num_action_types":
        self.selected_action_mask = mask.clone()

      # Agent makes a decision
      probs = agent.take_action(tree_trajectories, mask)
      selection = torch.argmax(probs).item()

      # Update our state representation
      tree_trajectories[-1][0, new_level] = selection

      if interpret_level[new_level] == "pattern_symbol":
        tree_trajectories.append(tree_trajectories[-1].clone())  # FIXME we may have extra tree trajectory at the end
        return None

      if self.verbose:
        print("On level:", new_level, "the agent chose", selection, "out of", cur_min, "through", cur_max)

      for iteration in range(1,selection+1):
        self.make_agent_selections(agent, tree_trajectories, cur_dict, new_level, iteration)

  def get_mechanic_dicts(self):
    dicts = dict()
    mechanics = self.mechanic_list
    if "Square-Grid Movement" in mechanics:
      dicts[mechanic_types["Square-Grid Movement"]] = self.square_dict()
    if "Betting" in mechanics:
      dicts[mechanic_types["Betting"]] = self.betting_dict()
    print("Dictionaries: \n%s" % str(dicts))
    return dicts


  # For these dictionaries, the min is always 1
  def square_dict(self):
    sq = {}
    # Key[level]            (Min, Max)
    sq["num_groups"] = (1, 1)  # num_group (how many of that mechanic there is)
    # Should record which group we are on
    sq["num_child_entities"] = (1, 6)  # num child entity types
    # Should record child entity type we are on
    sq["num_action_types"] = (1, 4)  # num_action_types
    sq["num_patterns"] = (1, 2)  # num_patterns
    # Should record which pattern we are looking at
    sq["pattern_length"] = (1, 3)  # pattern_length
    sq["pattern_symbol"] = (1, 9)  # pattern_symbol
    sq["num_parent_entity_types"] = 1
    return sq

  # For these dictionaries, the min is always 1
  def betting_dict(self):
    sq = {}
    # Key[level]            (Min, Max)
    sq["num_groups"] = (1, 1)  # num_group (how many of that mechanic there is)
    # Should record which group we are on
    sq["num_child_entities"] = (2, 2)  # num child entity types
    # Should record child entity type we are on
    sq["num_action_types"] = (1, 2)  # num_action_types
    sq["num_patterns"] = (1, 1)  # num_patterns
    # Should record which pattern we are looking at
    sq["pattern_length"] = (1, 1)  # pattern_length
    sq["pattern_symbol"] = (1, 1)  # pattern_symbol
    sq["num_parent_entity_types"] = 2
    return sq
  # [ 1 0 0 0  ... 0 0 0 0 ]
  # [ 0 1 0 0  ... 0 0 0 0 ]
  # [ 1 0 0 0  ][ 1 0 0 0  ]
  #

  # def capture_dict(self):
  #   sq = {}
  #   # Key[level]            (Min, Max)
  #   sq[1] =                 (1, 1)  # num_grids
  #   sq[2] =                 (1, 1)  # square_types
  #   sq[3] =                 (1, 6)  # piece_types
  #   sq[4] =                 (1, 2)  # num_patterns
  #   sq[5] =                 (1, 3)  # pattern_length
  #   sq[6] =                 (9, 9)  # pattern_symbols
  #   return sq

if __name__=='__main__':

  mechanic_list = ["Square-Grid Movement", "Betting"]
  game = Game(mechanic_list)
  agent = CreatorAgent()
  child_embeddings, child_trajectories, parent_embeddings, parent_trajectories = game.generate_entity_states(agent)
  print()