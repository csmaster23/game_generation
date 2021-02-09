import gym
import numpy as np
from gym import spaces

class Game(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, mechanic_list):
    super(Game, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    N_DISCRETE_ACTIONS = 0
    N_MECHANICS = 2
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
    for dict in mechanic_dicts:
      mechanic_widths.append(np.prod([dict[key][1] for key in dict]))
      num_levels.append(len(dict.keys()))
    width = max(mechanic_widths)
    height = max(num_levels)

    # Create the entity states based off the max and min values
    entity_states = np.zeros((len(mechanic_dicts), width, height))

    return entity_state, max_dict

  def get_mechanic_dicts(self):
    dicts = []
    mechanics = self.mechanic_list
    if "Square-Grid Movement" in mechanics:
      dicts.append( self.square_dict() )
    if "Static Capture" in mechanics:
      dicts.append( self.capture_dict() )
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
    sq[0] =                 (1, 1)  # num_grids
    sq[1] =                 (1, 1)  # square_types
    sq[2] =                 (1, 6)  # piece_types
    sq[3] =                 (1, 2)  # num_patterns
    sq[4] =                 (1, 3)  # pattern_length
    sq[5] =                 (9, 9)  # pattern_symbols
    return sq

if __name__=='__main__':

  mechanic_list = ["Square-Grid Movement", "Static Capture"]
  game = Game(mechanic_list)
  agent = None
  state = game.generate_entity_states(agent)