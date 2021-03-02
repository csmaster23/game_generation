import sys
import uuid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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

class Entity():
    def __init__(self, entity_names=None, parent_names=None, parents_to_actions=None, actions_to_patterns=None, id=None, p=None):
        """
        :param entity_names: List of names for of each piece type that the entity belongs to. Each name includes a generic
        name (from the mechanic) and a subtype.
        :param parent_names: Names of the parents of this entity. This specifies which entity types it can move across.
        Each name includes a generic name (from the mechanic) and a group number.
        :param parents_to_actions: A dictionary mapping a parent name to a list of action types (strings from mechanic)
        :param actions_to_patterns: A dictionary mapping an action type (string) to an action pattern (a sequence of entity
        group actions)
        :param id: An id that can be used to identify this distinct entity in the future
        :param p: A dictionary of parameters to fill in any more information
        """
        self.p = p
        self.entity_names = set() if entity_names is None else entity_names
        self.parent_names = set() if parent_names is None else parent_names
        # self.parents_to_actions = dict() if parents_to_actions is None else parents_to_actions
        self.actions_to_patterns = dict() if actions_to_patterns is None else actions_to_patterns
        self.info = None
        if id is None:
            self.id = uuid.uuid1()
        else:
            self.id = id
        self.storage_location = None
        self.my_stored_ids = []
        self.grid_wdith = 8 # really makes 5x5 grid
        self.original_piece_position = (3.5, 3.5) # middle of 5x5 grid, xy_coords
        self.current_piece_position = (3.5, 3.5)
        self.star_color = 'black'
        self.hop_color = 'purple'
        self.hop_to_color = 'blue'
        self.hop_over_color = 'navajowhite'
        self.drag_color = 'cyan'
        self.reverse_color = 'darkorange'
        self.match_color = 'darkblue'
        self.increment_color = 'darkgreen'
        self.card_move_list = ["reverse_turn", "match", "increment_draw"]

        self.entity_groups = []

    def show(self):
        print("Here in show!")
        # dict_keys(['p', 'entity_names', 'parent_names', 'actions_to_patterns', 'info', 'id', 'storage_location', 'my_stored_ids'])
        print(self.__dict__)
        for key in self.actions_to_patterns.keys():
            print("KEY: %s" % str(key))
            if key not in self.card_move_list:
                continue
            label_, obj_color, edge_color = self.get_label_and_colors(key)
            for i, move_key in enumerate(self.actions_to_patterns[key].keys()):
                self.current_piece_position = self.original_piece_position
                print("MOVE KEY: %s" % str(move_key))
                move_list = self.actions_to_patterns[key][move_key]
                # move_list = move_list + ['*']
                if key in self.card_move_list:
                    self.fig, self.ax = self.card_setup()
                else:
                    self.fig, self.ax = self.get_default_grid(key)
                for i, move_name in enumerate(move_list):
                    print("MOVE NAME: %s" % str(move_name))
                    self.add_plot_update_position(move_name, obj_color, edge_color, label_)

                piece = plt.Circle( self.original_piece_position, .3, facecolor='red', edgecolor='red', linewidth=3)
                self.ax.add_patch(piece)
                if key in self.card_move_list:
                    pass
                else:
                    plt.legend()
                plt.show()
        return

    def update_curr_position(self, x, y):
        self.current_piece_position = (self.current_piece_position[0] + x, self.current_piece_position[1] + y)

    def create_color_card(self, card_color, obj_color, name_):
        title_ = "Card Type: " + name_
        plt.title(title_)
        self.ax.add_patch(Rectangle((.2, 0), .6, 1, facecolor=obj_color, edgecolor='none', linewidth=5, zorder=1))
        self.ax.text(.35, .45, name_, fontsize=15, color='white', zorder=5)
    def add_num_to_card(self, card_num):
        title_ = "Card Number: " + card_num
        plt.title(title_)
        self.ax.add_patch(Rectangle((.2, 0), .6, 1, facecolor='black', edgecolor='navajowhite', linewidth=5, zorder=1))
        self.ax.text(.45, .45, card_num, fontsize=45, color='white', zorder=5)

    def add_arrow(self, x, y, x_len, y_len, line_color_, edge_color_, label_):
        plt.arrow( x, y, x_len, y_len, head_width=0.2, width=0.05, facecolor=line_color_, edgecolor=edge_color_, label=label_ )
    def add_star(self, x, y, color_, label_):
        plt.plot( x, y, marker='*', markerfacecolor=color_, markersize=15, markeredgecolor=color_, label=label_)

    def get_default_grid(self, key):
        fig = plt.figure()
        ax = fig.gca()
        ax.set_xticks(np.arange(0, self.grid_wdith, 1))
        ax.set_yticks(np.arange(0, self.grid_wdith, 1))
        ax.set_xlim([0, self.grid_wdith - 1])
        ax.set_ylim([0, self.grid_wdith - 1])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.title(self.get_title(key))
        plt.grid()
        return fig, ax
    def card_setup(self):
        fig = plt.figure()
        ax = fig.gca()
        ax.set_xticks(np.arange(0, 1, 1))
        ax.set_yticks(np.arange(0, 1, 1))
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        # plt.title(self.get_title(key))
        plt.grid()
        return fig, ax

    def get_title(self, key):
        string_builder = "Plot for "
        for s in self.entity_names:
            string_builder = string_builder + " " + s
        string_builder = string_builder + " Move: " + key
        return string_builder

    def add_plot_update_position(self, move_name, obj_color, edge_color, label_):
        z = 0.3
        # ------- SQUARES -------
        if 'N' == move_name:
            x, y = (0, 1)
            self.add_arrow(self.current_piece_position[0], self.current_piece_position[1], x, y - z, obj_color, edge_color, label_)
            self.update_curr_position(x, y)
        elif 'E' == move_name:
            x, y = (1, 0)
            self.add_arrow(self.current_piece_position[0], self.current_piece_position[1], x - z, y, obj_color, edge_color, label_)
            self.update_curr_position(x, y)
        elif 'S' == move_name:
            x, y = (0, -1)
            self.add_arrow(self.current_piece_position[0], self.current_piece_position[1], x, y + z, obj_color, edge_color, label_)
            self.update_curr_position(x, y)
        elif 'W' == move_name:
            x, y = (-1, 0)
            self.add_arrow(self.current_piece_position[0], self.current_piece_position[1], x + z, y, obj_color, edge_color, label_)
            self.update_curr_position(x, y)
        elif 'NE' == move_name:
            x, y = (1, 1)
            self.add_arrow(self.current_piece_position[0], self.current_piece_position[1], x - z, y - z, obj_color, edge_color, label_)
            self.update_curr_position(x, y)
        elif 'SE' == move_name:
            x, y = (1, -1)
            self.add_arrow(self.current_piece_position[0], self.current_piece_position[1], x - z, y + z, obj_color, edge_color, label_)
            self.update_curr_position(x, y)
        elif 'NW' == move_name:
            x, y = (-1, 1)
            self.add_arrow(self.current_piece_position[0], self.current_piece_position[1], x + z, y - z, obj_color, edge_color, label_)
            self.update_curr_position(x, y)
        elif 'SW' == move_name:
            x, y = (-1, -1)
            self.add_arrow(self.current_piece_position[0], self.current_piece_position[1], x + z, y + z, obj_color, edge_color, label_)
            self.update_curr_position(x, y)
        elif '*' == move_name:
            self.add_star(self.current_piece_position[0], self.current_piece_position[1], self.star_color, 'Star')
        # ------- SQUARES -------
        # ------- CARDS ------
        elif 'black' == move_name:
            self.create_color_card( 'black', obj_color, label_)
        elif 'red' == move_name:
            self.create_color_card( 'red', obj_color, label_)
        elif 'blue' == move_name:
            self.create_color_card( 'blue', obj_color, label_)
        elif 'green' == move_name:
            self.create_color_card( 'green', obj_color, label_)
        elif move_name.isnumeric(): # card number
            self.add_num_to_card( move_name )
        # ------- CARDS ------


        # 1: "black", 2: "red", 3: "blue", 4: "green", 5: "1", 6: "2", 7: "3", 8: "4", 9: "5", 10: "6", 11: "7",
        # 12: "8", 13: "9", 14: "10", 15: "11", 16: "12", 17: "13"

    def get_label_and_colors(self, key):
        label_ = ""
        # ------- SQUARES -------
        if "hop_to" in key:
            obj_color = self.hop_to_color
            label_ = label_ + "Hop To"
        elif "hop_over" in key:
            obj_color = self.hop_over_color
            label_ = label_ + "Hop Over"
        elif "hop" in key:
            obj_color = self.hop_color
            label_ = label_ + "Hop"
        elif "drag" in key:
            obj_color = self.drag_color
            label_ = label_ + "Drag"
        # ------- SQUARES -------
        # ------- CARDS ------
        elif "reverse_turn" in key:
            obj_color = self.reverse_color
            label_ = label_ + "Reverse Turn Card"
        elif "match" in key:
            obj_color = self.match_color
            label_ = label_ + "Match Cards"
        elif "increment_draw" in key:
            obj_color = self.increment_color
            label_ = label_ + "Increment Draw Card"
        # ------- CARDS ------
        else:
            obj_color = 'black'
            label_ = "THE LABEL GOT MESSED UP"
        # ------- CAPTURE -------
        if "capture" in key:  # we change the edge color for capture
            edge_color = 'limegreen'
            label_ = label_ + " Capture"
        else:
            edge_color = 'none'
        # ------- CAPTURE -------
        return label_, obj_color, edge_color

