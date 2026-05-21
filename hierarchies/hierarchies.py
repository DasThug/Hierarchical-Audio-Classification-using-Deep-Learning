from .hierarchyClass import HierarchyTree

# ------- Define hierarchy trees for datasets -------
# Define function as format: def build_{dataset_name}_tree(), which returns a HierarchyTree object for that dataset

def build_urbansound8k_tree():

    tree = HierarchyTree()

    # Level 0
    tree.add_child_node("root", "human_animal", level_idx=0)
    tree.add_child_node("root", "vehicle", level_idx=1)
    tree.add_child_node("root", "mechanical", level_idx=2)
    tree.add_child_node("root", "music", level_idx=3)

    # Level 1
    tree.add_child_node("human_animal", "human", level_idx=0)
    tree.add_child_node("human_animal", "animal", level_idx=1)

    tree.add_child_node("vehicle", "vehicle_operation", level_idx=2)
    tree.add_child_node("vehicle", "vehicle_signal", level_idx=3)

    tree.add_child_node("mechanical", "construction", level_idx=4)
    tree.add_child_node("mechanical", "ventilation", level_idx=5)
    tree.add_child_node("mechanical", "signal", level_idx=6)

    tree.add_child_node("music", "recorded", level_idx=7)

    # Level 2
    tree.add_child_node("ventilation", "air_conditioner", level_idx=0)
    tree.add_child_node("vehicle_operation", "car_horn", level_idx=1)
    tree.add_child_node("human", "children_playing", level_idx=2)
    tree.add_child_node("animal","dog_bark",level_idx=3)
    tree.add_child_node("construction","drilling",level_idx=4)
    tree.add_child_node("vehicle_operation","engine_idling",level_idx=5)
    tree.add_child_node("signal","gun_shot",level_idx=6)
    tree.add_child_node( "construction","jackhammer",level_idx=7)
    tree.add_child_node("vehicle_signal","siren",level_idx=8)
    tree.add_child_node("recorded","street_music",level_idx=9)


    tree.finalize()

    return tree


# ---------------------------------------------------


# Retrieval function, when importing file:
def get_hierarchy_tree(hierarchy_name):

    hierarchy_name = hierarchy_name.lower()

    if hierarchy_name == "urbansound8k":
        return build_urbansound8k_tree()

    raise ValueError(
        f"No hierarchy tree defined for hierarchy "
        f"{hierarchy_name!r}"
    )