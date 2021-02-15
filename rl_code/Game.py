import gym
import numpy as np
from gym import spaces
from rl_code.Agent import RandomAgent, CreatorAgent
from torch import nn
import torch

mechanic_types = {
  "Square-Grid Movement" : 1,
  "Betting"       : 2,
}

class Game(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, mechanic_list, verbose=True):
    super(Game, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    N_DISCRETE_ACTIONS = 0
    self.num_possible_mechanics = 2
    self.max_level = 8
    self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
    # Example for using image as input:
    HEIGHT = 0
    WIDTH = 0
    N_CHANNELS = 0
    self.observation_space = spaces.Box(low=0, high=255, shape=
                    (HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
    self.mechanic_list = mechanic_list
    self.verbose=verbose

  def step(self, action):
    # Execute one time step within the environment
    r = self.simulate()
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


  def generate_entities(self, entity_states, max_dict):
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


  def generate_entity_states(self, agent):
    """
    :param agent:
    :return:
    entity_embeddings (N, dim): A tensor of entity embeddings
    grouped_tree_trajectories (N, ?, max_level+1): A list of tensors with tree trajectories that describe how to create the different entities
    """
    state = []
    mechanic_dicts = self.get_mechanic_dicts()
    mechanic_widths, num_levels = [], []

    #
    # # Create the entity states based off the max and min values
    # entity_state = np.zeros((len(mechanic_dicts), width, height))
    # max_dict = 0
    # return entity_state, max_dict

    # First generated the max dict that breaks down the entity_states
    max_dict = {0: len(mechanic_dicts)}
    for level in range(1,self.max_level+1):
      maxxes = []
      for dict in mechanic_dicts.values():
        maxxes.append(dict[level][1])
      max_dict[level] = max(maxxes) # + 1 # We need to account for the choice of zero

    # We have a list of embeddings that represent the leaves of the tree. This is the current state
    # Are converted to embeddings before they are fed into the transformer
    tree_trajectories = [torch.LongTensor([[0]*9])]

    # Get the empty entity state matrix
    # height = self.max_level + 1
    width = np.prod([max_dict[key] for key in max_dict])
    # entities_state = np.zeros((height, width))

    # Make sure we have enough slots to encode the mechanic types
    # assert len(entities_state[0]) // len(mechanic_dicts) >= self.num_possible_mechanics

    entity_markers = []
    # Run the RL Agent
    for i, mechanic in enumerate(self.mechanic_list):
      # row_start = i * (width // max_dict[0])
      # row_stop = (i+1) * (width // max_dict[0])

      tree_trajectories[-1][0, 0] = mechanic_types[mechanic]
      # entities_state[0, row_start + mechanic_types[mechanic]] = 1
      cur_dict = mechanic_dicts[mechanic]

      # Recurse down until we have made all the selections we need
      self.make_agent_selections(agent, tree_trajectories, cur_dict, max_dict, 0, entity_markers)

    tree_trajectories = torch.stack(tree_trajectories)

    if self.verbose:
      print(tree_trajectories)

    # Extract entity embeddings
    entity_embeddings = self.get_entity_embeddings(agent, tree_trajectories, entity_markers)

    # Will tell us how to create the actual entity objects later
    grouped_tree_trajectories = self.get_grouped_tree_trajectories(tree_trajectories, entity_markers)

    return entity_embeddings, grouped_tree_trajectories

  def get_grouped_tree_trajectories(self, tree_trajectories, entity_markers):
    grouped_tree_trajectories = []
    for i, idx in enumerate(entity_markers):
      if i == len(entity_markers) - 1:
        stop = -1
      else:
        stop = entity_markers[i + 1]
      grouped_tree_trajectories.append(tree_trajectories[entity_markers[i]:stop].reshape(-1,self.max_level+1))
    return grouped_tree_trajectories

  def get_entity_embeddings(self, agent, tree_trajectories, entity_markers):
    entity_embeddings = []
    N = tree_trajectories.shape[0]
    with torch.no_grad():
      tree_embeddings = agent.gen_embedder(tree_trajectories).reshape(N, -1)
      for i, idx in enumerate(entity_markers):
        if i == len(entity_markers) - 1:
          stop = -1
        else:
          stop = entity_markers[i+1]
        entity_embedding = tree_embeddings[entity_markers[i]:stop].mean(0)
        unique_id = agent.gen_embedder(torch.LongTensor([[i+1]])).reshape(-1)
        entity_embeddings.append(torch.cat([entity_embedding, unique_id]))

    return torch.stack(entity_embeddings)

  def make_agent_selections(self, agent, tree_trajectories, cur_dict, max_dict, level, entity_markers):
    # if not np.any(entities_state[level, row_start:row_stop]):
    #   print('fail 1')
    #   return None

    new_level = level + 1
    # width = (row_stop - row_start)
    cur_min, cur_max = cur_dict[new_level]

    # Need to zero out all entries so transformer knows where we are on the tree
    if tree_trajectories[-1][0, new_level] != 0:
      tree_trajectories[-1][0, new_level:] = 0

    # Agent makes a decision
    selection = agent.take_action(tree_trajectories, (cur_min, cur_max))
    selection = int(selection[0].item())

    # Update our state representation
    tree_trajectories[-1][0, new_level] = selection
    if new_level == 4:
      entity_markers.append(len(tree_trajectories)-1) # FIXME Should probably run some tests on this

    # entities_state[new_level, row_start + selection - 1] = 1

    if self.verbose:
      print("On level:", new_level, "the agent chose", selection, "out of", cur_min, "through", cur_max, "with a max of",
            max_dict[new_level])

    if new_level == self.max_level:
      tree_trajectories.append(tree_trajectories[-1].clone())
      return None

    for i in range(selection):
      # new_row_start = row_start + i * (width // max_dict[new_level])
      # new_row_stop = row_start + (i + 1) * (width // max_dict[new_level])
      self.make_agent_selections(agent, tree_trajectories, cur_dict, max_dict, new_level, entity_markers)

# [ 0 0  0 0  0 0 ] 1 option level 0
# [ 0 0  0 0  0 0 ] 3 options level 1
# [ 0 0][0 0][0 0 ] 2 options level 2


  def get_mechanic_dicts(self):
    dicts = dict()
    mechanics = self.mechanic_list
    if "Square-Grid Movement" in mechanics:
      dicts["Square-Grid Movement"] = self.square_dict()
    if "Betting" in mechanics:
      dicts["Betting"] = self.betting_dict()
    print("Dictionaries: \n%s" % str(dicts))
    return dicts

  # For these dictionaries, the min is always 1
  def square_dict(self):
    sq = {}
    # Key[level]            (Min, Max)
    sq[1] =                 (1, 1)  # num_grids
    sq[2] =                 (1, 1)  # square_types
    sq[3] =                 (1, 6)  # piece_types
    sq[4] =                 (1, 3)  # num_action_types
    sq[5] =                 (1, 4)  # selected_action_type
    sq[6] =                 (1, 2)  # num_patterns
    sq[7] =                 (1, 3)  # pattern_length
    sq[8] =                 (1, 9)  # pattern_symbols
    return sq

  # For these dictionaries, the min is always 1
  def betting_dict(self):
    sq = {}
    # Key[level]            (Min, Max)
    sq[1] = (1, 1)  # num_grids
    sq[2] = (1, 4)  # square_types
    sq[3] = (1, 3)  # piece_types
    sq[4] = (1, 3)  # num_action_types
    sq[5] = (1, 2)  # selected_action_type
    sq[6] = (1, 1)  # num_patterns
    sq[7] = (1, 1)  # pattern_length
    sq[8] = (1, 1)  # pattern_symbols
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
  state, trajectories = game.generate_entity_states(agent)
  print()