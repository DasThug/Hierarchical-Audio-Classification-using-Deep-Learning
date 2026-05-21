# hierarchyClass.py

from collections import defaultdict
from typing import Literal

class HierarchyNode:
    """
    A single node in a hierarchy tree.

    Each node has:
    - name: unique string identifier
    - level: automatically inferred hierarchy depth
    - level_idx: class index within its level
    - parent: parent node
    - children: list of child nodes
    """

    def __init__(self, name, level, level_idx=None, parent=None):
        self.name = name
        self.level = level
        self.level_idx = level_idx
        self.parent = parent
        self.children = []

    def add_child(self, child_node):
        child_node.parent = self
        self.children.append(child_node)

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return len(self.children) == 0

    def path_nodes(self):
        node = self
        path = []

        while node is not None:
            if not node.is_root():
                path.append(node)
            node = node.parent

        return list(reversed(path))

    def path(self, output="names"):
        """
        Return path from root to this node.

        output:
            "nodes"   -> [HierarchyNode, ...]
            "names"   -> [name, ...]
            "indices" -> [level_idx, ...]
            "pairs"   -> [(level, level_idx, name), ...]
        """
        nodes = self.path_nodes()

        if output == "nodes":
            return nodes

        if output == "names":
            return [node.name for node in nodes]

        if output == "indices":
            return [node.level_idx for node in nodes]

        if output == "pairs":
            return [
                (node.level, node.level_idx, node.name)
                for node in nodes
            ]

        raise ValueError(
            f"Invalid output={output!r}. "
            "Expected 'nodes', 'names', 'indices', or 'pairs'."
        )

    def child_path(self, output="names"):
        """
        Return immediate children.

        output:
            "nodes"   -> [HierarchyNode, ...]
            "names"   -> [name, ...]
            "indices" -> [level_idx, ...]
            "pairs"   -> [(level, level_idx, name), ...]
        """
        children = sorted(self.children, key=lambda node: node.level_idx)

        if output == "nodes":
            return children

        if output == "names":
            return [child.name for child in children]

        if output == "indices":
            return [child.level_idx for child in children]

        if output == "pairs":
            return [
                (child.level, child.level_idx, child.name)
                for child in children
            ]

        raise ValueError(
            f"Invalid output={output!r}. "
            "Expected 'nodes', 'names', 'indices', or 'pairs'."
        )

    def __repr__(self):
        return (
            f"HierarchyNode("
            f"name={self.name!r}, "
            f"level={self.level}, "
            f"level_idx={self.level_idx})"
        )


class HierarchyTree:
    """
    Generic hierarchy tree.

    Usage:
        tree = HierarchyTree()

        tree.add_child_node(
            target="root",
            name="human_animal",
            level_idx=0
        )

        tree.add_child_node(
            target="human_animal",
            name="human",
            level_idx=0
        )

        tree.finalize()
    
    Methods:
        - get_path(): params {(name:str) or (level:int + level_idx:int) or (leaf_idx:int)} + {output="names" or "indices" or "pairs"}
            -> returns hierarchical path as list: [str, str, ...] or [int, int, ...] or [(level, level_idx, name), ...], depending on output format
        
        - get_children(): params {(name:str) or (level:int + level_idx:int)} + {output="names" or "indices" or "pairs"}
            -> returns immediate children of the node, as list: [str, str, ...] or [int, int, ...] or [(level, level_idx, name), ...], depending on output format
        
        - get_level_idx_from_name(name:str) -> (level:int, level_idx:int)
        - get_name_from_level_idx(level:int, level_idx:int) -> name:str
        
        - get_level_class_to_idx() -> lookup dict {level: {class_name: level_idx}}
        - get_level_idx_to_class() -> lookup dict {level: {level_idx: class_name}}
        - path_indices_to_names(indices:list of int) -> names:list of str
        - path_names_to_indices(names:list of str) -> indices:list of int
        - num_classes_per_level() -> dict {level: num_classes}
    """

    def __init__(self, root_name="root"):
        self.root_name = root_name
        self.root = HierarchyNode(
            name=root_name,
            level=-1,
            level_idx=None,
            parent=None
        )

        self.nodes_by_name = {root_name: self.root}
        self.nodes_by_level = defaultdict(list)
        self.level_idx_to_name = defaultdict(dict)

        self.total_depth = 0
        self.closed = False

    def _assert_not_closed(self):
        if self.closed:
            raise RuntimeError(
                "HierarchyTree is finalized/closed. "
                "No more nodes can be added."
            )

    def add_child_node(self, target, name, level_idx):
        """
        Add a new child node to an existing parent node.

        Parameters
        ----------
        target : str
            Name of the parent node.

        name : str
            Name of the new child node.

        level_idx : int
            Index of the node within its inferred hierarchy level.

        Returns
        -------
        HierarchyNode
            The created node.
        """

        self._assert_not_closed()

        if target not in self.nodes_by_name:
            raise ValueError(f"Parent target {target!r} does not exist.")

        if name in self.nodes_by_name:
            raise ValueError(f"Node name {name!r} already exists.")

        if not isinstance(level_idx, int):
            raise TypeError(
                f"level_idx must be an int, got {type(level_idx)}."
            )

        parent = self.nodes_by_name[target]
        level = parent.level + 1

        if level_idx in self.level_idx_to_name[level]:
            existing_name = self.level_idx_to_name[level][level_idx]
            raise ValueError(
                f"Index {level_idx} is already used at level {level} "
                f"by node {existing_name!r}."
            )

        node = HierarchyNode(
            name=name,
            level=level,
            level_idx=level_idx,
            parent=parent
        )

        parent.add_child(node)

        self.nodes_by_name[name] = node
        self.nodes_by_level[level].append(node)
        self.level_idx_to_name[level][level_idx] = name
        self.total_depth = max(self.total_depth, level + 1)

        return node

    def get_node(self, name=None, level=None, level_idx=None, leaf_idx=None):
        """
        Flexible node lookup.

        Supported inputs:
            get_node("siren")
            get_node(name="siren")
            get_node(level=2, level_idx=8)
            get_node(leaf_idx=8)

        leaf_idx is interpreted as:
            index at the deepest hierarchy level.
        """

        if isinstance(name, HierarchyNode):
            return name

        if name is not None:
            if name not in self.nodes_by_name:
                raise KeyError(f"Node {name!r} does not exist.")
            return self.nodes_by_name[name]

        if leaf_idx is not None:
            level = self.total_depth - 1
            level_idx = leaf_idx

        if level is None or level_idx is None:
            raise ValueError(
                "Provide either name, leaf_idx, or both level and level_idx."
            )

        if level not in self.level_idx_to_name:
            raise ValueError(f"Level {level} does not exist.")

        if level_idx not in self.level_idx_to_name[level]:
            raise ValueError(
                f"Index {level_idx} does not exist at level {level}."
            )

        resolved_name = self.level_idx_to_name[level][level_idx]
        return self.nodes_by_name[resolved_name]

    def get_parent(self, name=None, level=None, level_idx=None, leaf_idx=None):
        return self.get_node(
            name=name,
            level=level,
            level_idx=level_idx,
            leaf_idx=leaf_idx
        ).parent

    def get_children(
        self,
        name=None,
        level=None,
        level_idx=None,
        output: Literal["nodes", "names", "indices", "pairs"] = "indices"
    ):
        """
        Return immediate children of a node.

        output:
            "nodes"   -> [HierarchyNode, ...]
            "names"   -> [name, ...]
            "indices" -> [level_idx, ...]
            "pairs"   -> [(level, level_idx, name), ...]
        """
        node = self.get_node(
            name=name,
            level=level,
            level_idx=level_idx,
        )

        return node.child_path(output=output)

    def get_path(
        self,
        name=None,
        level=None,
        level_idx=None,
        leaf_idx=None,
        output: Literal["names", "indices", "pairs"] = "names"
    ):
        """
        Return full hierarchy path to a node.

        output:
            "nodes"   -> [HierarchyNode, ...]
            "names"   -> [name, ...]
            "indices" -> [level_idx, ...]
            "pairs"   -> [(level, level_idx, name), ...]
        """
        node = self.get_node(
            name=name,
            level=level,
            level_idx=level_idx,
            leaf_idx=leaf_idx
        )

        return node.path(output=output)

    def get_path_names(self, *args, **kwargs):
        kwargs["output"] = "names"
        return self.get_path(*args, **kwargs)

    def get_path_indices(self, *args, **kwargs):
        kwargs["output"] = "indices"
        return self.get_path(*args, **kwargs)

    def get_path_pairs(self, *args, **kwargs):
        kwargs["output"] = "pairs"
        return self.get_path(*args, **kwargs)

    def get_level_nodes(self, level):
        return sorted(
            list(self.nodes_by_level[level]),
            key=lambda node: node.level_idx
        )

    def get_name_from_level_idx(self, level, level_idx):
        return self.get_node(level=level, level_idx=level_idx).name

    def get_level_idx_from_name(self, name):
        node = self.get_node(name=name)
        return node.level, node.level_idx

    def get_level_class_to_idx(self):
        """
        Returns:
            {
                level: {
                    class_name: level_idx
                }
            }
        """
        return {
            level: {
                node.name: node.level_idx
                for node in self.get_level_nodes(level)
            }
            for level in sorted(self.nodes_by_level.keys())
        }

    def get_level_idx_to_class(self):
        """
        Returns:
            {
                level: {
                    level_idx: class_name
                }
            }
        """
        return {
            level: {
                node.level_idx: node.name
                for node in self.get_level_nodes(level)
            }
            for level in sorted(self.nodes_by_level.keys())
        }

    def path_indices_to_names(self, indices):
        """
        Convert full hierarchy path indices into names.

        Example:
            [1, 3, 8]
            ->
            ['vehicle', 'vehicle_signal', 'siren']
        """
        names = []

        for level, idx in enumerate(indices):
            names.append(
                self.get_name_from_level_idx(level=level, level_idx=idx)
            )

        return names

    def path_names_to_indices(self, names):
        """
        Convert full hierarchy path names into indices.

        Example:
            ['vehicle', 'vehicle_signal', 'siren']
            ->
            [1, 3, 8]
        """
        indices = []

        for expected_level, name in enumerate(names):
            node = self.get_node(name=name)

            if node.level != expected_level:
                raise ValueError(
                    f"Node {name!r} is at level {node.level}, "
                    f"expected level {expected_level}."
                )

            indices.append(node.level_idx)

        return indices

    def num_classes_per_level(self):
        """
        Returns:
            {
                level: number_of_classes
            }
        """
        return {
            level: len(nodes)
            for level, nodes in sorted(self.nodes_by_level.items())
        }

    def leaf_nodes(self):
        return sorted(
            [
                node
                for node in self.nodes_by_name.values()
                if not node.is_root() and node.is_leaf()
            ],
            key=lambda node: (node.level, node.level_idx)
        )

    def leaf_names(self):
        return [node.name for node in self.leaf_nodes()]

    def validate(self):
        """
        Basic hierarchy validation.
        """

        if len(self.nodes_by_level) == 0:
            raise ValueError("Hierarchy contains no non-root nodes.")

        for level, nodes in self.nodes_by_level.items():
            seen_indices = set()

            for node in nodes:
                if node.level_idx in seen_indices:
                    raise ValueError(
                        f"Duplicate level_idx={node.level_idx} "
                        f"at level {level}."
                    )

                seen_indices.add(node.level_idx)

                if node.parent is None:
                    raise ValueError(
                        f"Non-root node {node.name!r} has no parent."
                    )

                if node.level != node.parent.level + 1:
                    raise ValueError(
                        f"Invalid level for node {node.name!r}. "
                        f"Expected {node.parent.level + 1}, "
                        f"got {node.level}."
                    )

        expected_depth = max(self.nodes_by_level.keys()) + 1

        if self.total_depth != expected_depth:
            raise ValueError(
                f"Invalid total_depth={self.total_depth}. "
                f"Expected {expected_depth}."
            )

        return True

    def finalize(self):
        """
        Validate and freeze the hierarchy.
        """
        self.validate()
        self.closed = True

    def is_finalized(self):
        return self.closed

    def print_tree(self):
        def recurse(node, indent=0):
            if node.is_root():
                print(node.name)
            else:
                print(
                    " " * indent
                    + f"- {node.name} "
                    + f"[level={node.level}, idx={node.level_idx}]"
                )

            for child in sorted(node.children, key=lambda n: n.level_idx):
                recurse(child, indent + 4)

        recurse(self.root)

    def __repr__(self):
        return (
            f"HierarchyTree("
            f"total_depth={self.total_depth}, "
            f"num_nodes={len(self.nodes_by_name) - 1}, "
            f"closed={self.closed})"
        )


if __name__ == "__main__":
    US_hierarchy_tree = HierarchyTree()

    # Level 0
    US_hierarchy_tree.add_child_node("root", "human_animal", level_idx=0)
    US_hierarchy_tree.add_child_node("root", "vehicle", level_idx=1)
    US_hierarchy_tree.add_child_node("root", "mechanical", level_idx=2)
    US_hierarchy_tree.add_child_node("root", "music", level_idx=3)

    # Level 1
    US_hierarchy_tree.add_child_node("human_animal", "human", level_idx=0)
    US_hierarchy_tree.add_child_node("human_animal", "animal", level_idx=1)

    US_hierarchy_tree.add_child_node("vehicle", "vehicle_operation", level_idx=2)
    US_hierarchy_tree.add_child_node("vehicle", "vehicle_signal", level_idx=3)

    US_hierarchy_tree.add_child_node("mechanical", "construction", level_idx=4)
    US_hierarchy_tree.add_child_node("mechanical", "ventilation", level_idx=5)
    US_hierarchy_tree.add_child_node("mechanical", "signal", level_idx=6)

    US_hierarchy_tree.add_child_node("music", "recorded", level_idx=7)

    # Level 2
    US_hierarchy_tree.add_child_node("ventilation", "air_conditioner", level_idx=0)
    US_hierarchy_tree.add_child_node("vehicle_operation", "car_horn", level_idx=1)
    US_hierarchy_tree.add_child_node("human", "children_playing", level_idx=2)
    US_hierarchy_tree.add_child_node("animal", "dog_bark", level_idx=3)
    US_hierarchy_tree.add_child_node("construction", "drilling", level_idx=4)
    US_hierarchy_tree.add_child_node("vehicle_operation", "engine_idling", level_idx=5)
    US_hierarchy_tree.add_child_node("signal", "gun_shot", level_idx=6)
    US_hierarchy_tree.add_child_node("construction", "jackhammer", level_idx=7)
    US_hierarchy_tree.add_child_node("vehicle_signal", "siren", level_idx=8)
    US_hierarchy_tree.add_child_node("recorded", "street_music", level_idx=9)

    US_hierarchy_tree.finalize()

    US_hierarchy_tree.print_tree()

    print()
    print("Total depth:", US_hierarchy_tree.total_depth)

    print()
    print("Path by name:")
    print(US_hierarchy_tree.get_path("siren", output="names"))
    print(US_hierarchy_tree.get_path("siren", output="indices"))
    print(US_hierarchy_tree.get_path("siren", output="pairs"))

    print()
    print("Path by leaf index:")
    print(US_hierarchy_tree.get_path(leaf_idx=8, output="names"))
    print(US_hierarchy_tree.get_path(leaf_idx=8, output="indices"))
    print(US_hierarchy_tree.get_path(leaf_idx=8, output="pairs"))

    print()
    print("Path by level/index:")
    print(US_hierarchy_tree.get_path(level=2, level_idx=8, output="names"))

    print()
    print("Children of vehicle:")
    print(US_hierarchy_tree.get_children("vehicle", output="names"))
    print(US_hierarchy_tree.get_children(level=0, level_idx=1, output="indices"))
    print(US_hierarchy_tree.get_children("vehicle", output="pairs"))

    print()
    print("Name/index lookup:")
    print(US_hierarchy_tree.get_level_idx_from_name("siren"))
    print(US_hierarchy_tree.get_name_from_level_idx(level=2, level_idx=8))

    print()
    print("Level mappings:")
    print(US_hierarchy_tree.get_level_class_to_idx())
    print(US_hierarchy_tree.get_level_idx_to_class())
    print(US_hierarchy_tree.num_classes_per_level())

    print()
    print("Test prints")
    print(US_hierarchy_tree.get_path(level=1, level_idx=1, output="pairs"))
