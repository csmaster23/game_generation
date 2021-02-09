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
    self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
    # Example for using image as input:
    HEIGHT = 0
    WIDTH = 0
    N_CHANNELS = 0
    self.observation_space = spaces.Box(low=0, high=255, shape=
                    (HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

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
  def generate_states(self, agent, mechanic_list):
    state = []
    mechanic_dicts = self.get_mechanic_dicts(mechanic_list)
    # ...
    # Jamison stub
    # ...
    return state

  def get_mechanic_dicts(self, mechanics):
    dicts = []
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
    # Key                   (Min, Max)
    sq["num_grids"] =       (1, 2)
    sq["squares"] =         (2, 5)
    sq["piece_types"] =     (1, 3)
    sq["num_patterns"] =    (1, 3)
    return sq

  def capture_dict(self):
    cp = {}
    # Key                   (Min, Max)
    cp["num_grids"] =       (1, 2)
    cp["squares"] =         (2, 5)
    cp["piece_types"] =     (1, 3)
    cp["num_patterns"] =    (1, 3)
    return cp