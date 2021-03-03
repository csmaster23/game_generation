
import sys


def plot_child_entities( entities ):
    child_entities = [ entities[key] for key in entities.keys() if len(entities[key].parent_names) > 0]
    [print(value.__dict__) for value in entities.values()]
    # sys.exit(0)
    for entity in child_entities:
        entity.show()
    # sys.exit(0)
    return
