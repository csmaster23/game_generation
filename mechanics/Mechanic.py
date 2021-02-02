from mechanics.Piece import Piece


# ---- Start/Base ----

# Entity - Board Locations, Cards, Pieces
    # List of other entities
    # Unique Id

# Entity Movement Rules
    # Describe how movement occurs
    # Adjacency Matrix for each entity

# Mechanisms
    # Creates entity
    # Creates board
    # Creates trackers
    # Adds Rounds to a game
        # list of player id's with turn order
    # Adds hard-coded actions such as
        #
    # Can create type and sub-types of pieces

# Player
    # Stores trackers owned by player
    # Selects and performs actions

# Round
    # returns player Id of who won that specific round
    # Method that returns winner for round

# Action
    # Takes in an action number and player id
    # Changes player tracker values
    # Actually moves entities according to entity rules created by mechanisms
    # Returns list of actions with enable/disable description
    # End-Turn action, ends round and starts with player 1 again




# Move Pieces, Player Trackers, Grid Locations

# Simulation Agent can perform actions
    # Actions Include
        # Moving Pieces
        # Game Specific Actions created by mechanisms
        # Determines which actions are enabled/disabled

# RL Agent
    # Combines Types of Pieces

# Piece
    # Unique-Id
    # Types
        # Sub-Types

# Types
    # Area Movement Piece
        # Sub-Type is Queen or Pawn
    # Capture
        # No sub-type



class Mechanic():
    def __init__(self):
        self.pieces = []


    def add_piece(self, piece):
        self.pieces.append(piece)