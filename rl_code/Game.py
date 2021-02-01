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
    pass
  def reset(self):
    # Reset the state of the environment to an initial state
    return 'resetted'
  def render(self, mode='human', close=False):
    # Render the environment to the screen
    return 'rendered'
  def simulate(self):
    pass