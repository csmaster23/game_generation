import gym
import numpy as np
from gym import spaces
from Agent import RandomAgent

mechanic_types = {
  "Square-Grid Movement" : 0,
  "Static Capture"       : 1,
}

class Game(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, mechanic_list):
    super(Game, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    N_DISCRETE_ACTIONS = 0
    self.num_possible_mechanics = 2
    self.max_level = 6
    self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
    # Example for using image as input:
    HEIGHT = 0
    WIDTH = 0
    N_CHANNELS = 0
    self.observation_space = spaces.Box(low=0, high=255, shape=
                    (HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
    self.mechanic_list = mechanic_list

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

  def generate_entity_states(self, agent):
    state = []
    mechanic_dicts = self.get_mechanic_dicts()
    mechanic_widths, num_levels = [], []

    # First generated the max dict that breaks down the entity_states
    max_dict = {0: len(mechanic_dicts)}
    for level in range(1,self.max_level+1):
      maxxes = []
      for dict in mechanic_dicts.values():
        maxxes.append(dict[level][1])
      max_dict[level] = max(maxxes) # + 1 # We need to account for the choice of zero
      if level == 6: # On each of these levels, we do not have the option to put zero
        max_dict[level] -= 1

    # Get the empty entity state matrix
    height = self.max_level + 1
    width = np.prod([max_dict[key] for key in max_dict])
    entities_state = np.zeros((height, width))

    # Make sure we have enough slots to encode the mechanic types
    assert len(entities_state[0]) // len(mechanic_dicts) >= self.num_possible_mechanics

    # Run the RL Agent
    for i, mechanic in enumerate(mechanic_list):
      row_start = i * (width // max_dict[0])
      row_stop = (i+1) * (width // max_dict[0])
      entities_state[0, row_start + mechanic_types[mechanic]] = 1
      cur_dict = mechanic_dicts[mechanic]

      # Recurse down until we have made all the selections we need
      self.make_agent_selections(agent, entities_state, cur_dict, max_dict, 0, row_start, row_stop, 1)

    return entities_state, max_dict

  def make_agent_selections(self, agent, entities_state, cur_dict, max_dict, level, row_start, row_stop, prev_choice):
    if not np.any(entities_state[level, row_start:row_stop]) or level == self.max_level:
      return None

    level += 1
    width = row_stop - row_start
    for i in range(prev_choice):
      new_row_start = row_start + i * (width // max_dict[level])
      new_row_stop = row_start + (i+1) * (width // max_dict[level])
      cur_min, cur_max = cur_dict[level]

      # Agent makes a decision
      selection = agent.take_action(cur_min, cur_max)

      # Update our state representation
      entities_state[level, new_row_start + selection] = 1
      print("On level:", level, "the agent chose", selection, "out of", cur_min, "through", cur_max, "with a max of", max_dict[level] - 1)
      print(entities_state[level, new_row_start:new_row_stop])
      self.make_agent_selections(agent, entities_state, cur_dict, max_dict, level, new_row_start, new_row_stop, selection)

# [ 0 0  0 0  0 0 ] 1 option level 0
# [ 0 0  0 0  0 0 ] 3 options level 1
# [ 0 0][0 0][0 0 ] 2 options level 2


  def get_mechanic_dicts(self):
    dicts = dict()
    mechanics = self.mechanic_list
    if "Square-Grid Movement" in mechanics:
      dicts["Square-Grid Movement"] = self.square_dict()
    if "Static Capture" in mechanics:
      dicts["Static Capture"] = self.capture_dict()
    print("Dictionaries: \n%s" % str(dicts))
    # .
    # .
    # .
    return dicts

  def square_dict(self):
    sq = {}
    # Key[level]            (Min, Max)
    sq[1] =                 (1, 1)  # num_grids
    sq[2] =                 (1, 1)  # square_types
    sq[3] =                 (1, 6)  # piece_types
    sq[4] =                 (1, 2)  # num_patterns
    sq[5] =                 (1, 3)  # pattern_length
    sq[6] =                 (9, 9)  # pattern_symbols
    return sq

  # [ 1 0 0 0  ... 0 0 0 0 ]
  # [ 0 1 0 0  ... 0 0 0 0 ]
  # [ 1 0 0 0  ][ 1 0 0 0  ]
  #

  def capture_dict(self):
    sq = {}
    # Key[level]            (Min, Max)
    sq[1] =                 (1, 1)  # num_grids
    sq[2] =                 (1, 1)  # square_types
    sq[3] =                 (1, 6)  # piece_types
    sq[4] =                 (1, 2)  # num_patterns
    sq[5] =                 (1, 3)  # pattern_length
    sq[6] =                 (9, 9)  # pattern_symbols
    return sq

if __name__=='__main__':

  mechanic_list = ["Square-Grid Movement", "Static Capture"]
  game = Game(mechanic_list)
  agent = RandomAgent()
  state = game.generate_entity_states(agent)
  print()