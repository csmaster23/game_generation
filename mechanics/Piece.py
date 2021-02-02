


class Piece():
    def __init__(self, owner):
        self.type = {}
        self.mechanics = []
        self.owner = owner


    def formulate_types(self, type_, sub_types):
        self.type[type_] = sub_types

    def piece_constraints(self, piece_name):
        pass