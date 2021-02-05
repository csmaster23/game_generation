from mechanics.Piece import Piece
from mechanics.Entity import Entity

# ---- Start/Base ----

# Entity Movement Rules
    # Describe how movement occurs
    # Adjacency Matrix for each entity

# Entity - Board Locations, Cards, Pieces
    # List of other entities
    # Unique Id
    # Contains Entity Movements Rules


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

# Round...?
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
    def __init__(self, p):
        self.p = p              # parameters
        self.entities = self.create_entities()

    def create_entities(self):
        return []

    def add_piece(self, piece):
        self.pieces.append(piece)


    def combine_layers(self):
        pass

    def generic_piece_entity(self, mechanic_type, num_pieces):
        entities = []
        for i in range(num_pieces):
            entities.append(Entity(self.p, mechanic_type))
        return entities

    def square_grid_method(self, entities_in):
        # define entity group from the squares
        # create square grid generator
        pass

    def static_capture_formalization(self, piece, flags, p):
        # flags = [ hop_capture, forward_capture, backward_capture, left_capture, right_capture, d_45_capture, d_135_capture, d_225_capture, d_315_capture ] Bools
        # Hop To Capture =
        # Hop Over Capture =
        # Drag Capture =
        for f in flags:
            piece.pattern_dict[f] = None

    def static_movement_formalization(self, piece, flags, p):
        # flags = [ hop_capture, forward_capture, backward_capture, left_capture, right_capture, d_45_capture, d_135_capture, d_225_capture, d_315_capture ] Bools
        pass

    def static_capture_piece_init(self, pieces, pattern_type, move_tuple):
        for piece in pieces:
            piece



    def static_capture_pattern(self, pattern, num_tuple, p):
        pass


    # hop = 2*, 4*, 5*, 7*
    # drag = 1*,....8*