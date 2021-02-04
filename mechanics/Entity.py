

class Entity():
    def __init__(self, entity_type, id, p):
        self.p = p
        self.entity_type = entity_type # string e.x. Square_Grid
        self.id = id
        self.my_store_types = []
        self.my_storage_location = 0
        self.my_stored_ids = []