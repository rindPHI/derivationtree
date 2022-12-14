# Copyright © 2022 CISPA Helmholtz Center for Information Security.
# Author: Dominic Steinhöfel.
#
# This file is part of ISLa.
#
# ISLa is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ISLa is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ISLa.  If not, see <http://www.gnu.org/licenses/>.

import html
import itertools
import json
import re
import zlib
from dataclasses import dataclass
from functools import lru_cache
from typing import (
    Tuple,
    List,
    Optional,
    Dict,
    Iterable,
    Set,
    Generator,
    Callable,
    Union,
    Sequence,
    cast,
)

import datrie
from grammar_graph import gg
from graphviz import Digraph

ParseTree = Tuple[str, Optional[List["ParseTree"]]]

next_id = 0


def get_next_id() -> int:
    global next_id
    next_id += 1
    return next_id


@dataclass(frozen=True)
class DerivationTreeNode:
    node_id: int
    value: str


class MemoMeta(type):
    def __init__(self, name, bases, namespace):
        super().__init__(name, bases, namespace)
        self.cache = {}

    def __call__(self, *args):
        if args not in self.cache:
            self.cache[args] = super().__call__(*args)
        return self.cache[args]


class Path(metaclass=MemoMeta):
    def __init__(self, *args):
        assert (
            len(args) == 1
            and isinstance(args[0], str)
            or len(args) == 1
            and isinstance(args[0], tuple)
            and all(isinstance(elem, int) for elem in args[0])
            or all(isinstance(elem, int) for elem in args)
        )

        key = (
            chr(1)
            if not args
            else (
                args[0]
                if isinstance(args[0], str) or isinstance(args[0], tuple)
                else args
            )
        )

        assert not isinstance(key, str) or key and key[0] == chr(1)

        self.__key = (
            key
            if isinstance(key, str)
            else chr(1) + "".join([chr(i + 2) for i in cast(Tuple[int, ...], key)])
        )

    def key(self) -> str:
        return self.__key

    def startswith(self, other: "Path") -> bool:
        return self.__key.startswith(other.__key)

    def __getitem__(self, idx: int | slice) -> Union[int, "Path"]:
        assert isinstance(idx, int) or isinstance(idx, slice)

        if isinstance(idx, int):
            if not -len(self.__key) + 1 <= idx < len(self.__key) - 1:
                raise IndexError

            idx = idx + 1 if idx >= 0 else len(self.__key) + idx

            return ord(self.__key[idx]) - 2
        else:
            elems = self.__key[1:][idx]
            return Path(chr(1) + elems)

    def __iter__(self):
        class PathIterator:
            def __init__(self, key: str):
                self.__idx = 0
                self.__key = key

            def __next__(self):
                self.__idx += 1

                if self.__idx - 1 < len(self.__key) - 1:
                    return ord(self.__key[self.__idx]) - 2
                else:
                    raise StopIteration

            def __getitem__(self, idx: int) -> int:
                assert isinstance(idx, int)
                idx = idx + 1 if idx >= 0 else len(self.__key) + idx
                return ord(self.__key[self.__idx + idx]) - 2

        return PathIterator(self.__key)

    def __len__(self):
        return len(self.__key) - 1

    def __bool__(self) -> bool:
        return len(self.__key) > 1

    def __add__(self, other: Union["Path", Tuple[int, ...], str]) -> "Path":
        assert (
            isinstance(other, Path)
            or isinstance(other, tuple)
            and all(isinstance(elem, int) for elem in other)
            or isinstance(other, str)
        )
        return (
            Path(self.__key + other.__key[1:])
            if isinstance(other, Path)
            else (
                Path(self.__key + other)
                if isinstance(other, str)
                else Path(self.__key + "".join([chr(elem + 2) for elem in other]))
            )
        )

    def __repr__(self):
        return f'Path("{self.__key}")'

    def __str__(self):
        path = (
            ()
            if self.__key == chr(1)
            else tuple([ord(c) - 2 for c in self.__key if ord(c) != 1])
        )
        return str(path)

    def __eq__(self, other: "Path") -> bool:
        if not isinstance(other, Path):
            return False

        return self.__key == other.__key

    def __hash__(self):
        return hash(self.__key)


class DerivationTree:
    TRAVERSE_PREORDER = 0
    TRAVERSE_POSTORDER = 1

    def __init__(
        self,
        init_map: Optional[Dict[Path, DerivationTreeNode | str]] = None,
        init_trie: Optional[datrie.Trie] = None,
        open_leaves: Optional[Iterable[Path]] = None,
        root_path: Optional[Path] = Path(),
    ):
        assert init_map or init_trie

        self.__trie = DerivationTree.__trie_from_init_map(init_map, init_trie)
        self.__root_path = root_path

        self.__open_leaves: Set[Path] = {
            path for path in open_leaves or [] if path.startswith(self.__root_path)
        }

        self.__k_paths: Dict[int, Set[Tuple[gg.Node, ...]]] = {}
        self.__concrete_k_paths: Dict[int, Set[Tuple[gg.Node, ...]]] = {}

    @staticmethod
    def from_node(
        value: str, node_id: Optional[int] = None, is_open: Optional[bool] = None
    ):
        if is_open is None:
            is_open = is_nonterminal(value)

        return DerivationTree(
            {Path(): DerivationTreeNode(node_id or get_next_id(), value)},
            open_leaves={Path()} if is_open else set(),
        )

    @staticmethod
    def __trie_from_init_map(
        init_map: Optional[Dict[Path, DerivationTreeNode | str]] = None,
        init_trie: Optional[datrie.Trie] = None,
    ):
        if init_trie is not None:
            return init_trie

        result: datrie.Trie = datrie.Trie([chr(i) for i in range(32)])
        for path, node_or_value in (init_map or {}).items():
            result[path.key()] = (
                node_or_value
                if isinstance(node_or_value, DerivationTreeNode)
                else DerivationTreeNode(get_next_id(), node_or_value)
            )

        return result

    def node(self, path: Path = Path()) -> DerivationTreeNode:
        return self.__trie[self.__to_absolute_key(path).key()]

    def node_id(self, path: Path = Path()) -> int:
        return self.__trie[self.__to_absolute_key(path).key()].node_id

    def value(self, path: Path = Path()) -> str:
        return self.__trie[self.__to_absolute_key(path).key()].value

    def children(self, path: Path = Path()) -> Optional[Dict[Path, DerivationTreeNode]]:
        if self.is_open(path):
            return None

        if self.is_leaf(path):
            return {}

        result: Dict[Path, DerivationTreeNode] = {}

        i = 0
        while True:
            try:
                child_path = path + Path(i)
                result[child_path] = self.__trie[
                    self.__to_absolute_key(child_path).key()
                ]
                i += 1
            except KeyError:
                break

        return result

    def is_leaf(self, path: Path = Path()) -> bool:
        return (
            self.__trie.get(self.__to_absolute_key(path).key() + "\x02", None) is None
        )

    def is_open(self, path: Path = ()) -> bool:
        return self.__to_absolute_key(path) in self.__open_leaves

    def is_complete(self) -> bool:
        return not self.tree_is_open()

    def tree_is_open(self):
        return bool(self.__open_leaves)

    def is_valid_path(self, path: Path) -> bool:
        return self.__to_absolute_key(path).key() in self.__trie

    def paths(self) -> Generator[Tuple[Path, DerivationTreeNode], None, None]:
        """
        Returns a mapping from paths in this derivation tree to the corresponding
        DerivationTreeNode.

        :return: A mapping from paths to nodes.
        """

        state = datrie.State(self.__trie)
        state.walk(self.__root_path.key())
        it = datrie.Iterator(state)
        while it.next():
            yield Path(chr(1) + it.key()), it.data()

        # return {
        #     Path(chr(1) + suffix): self.__trie[(self.__root_path + suffix).key()]
        #     for suffix in self.__trie.suffixes(self.__root_path.key())
        # }

    def leaves(self) -> Generator[Tuple[Path, DerivationTreeNode], None, None]:
        def is_leaf(key: Path) -> bool:
            return self.__trie.get((key + (0,)).key(), None) is None

        state = datrie.State(self.__trie)
        state.walk(self.__root_path.key())
        it = datrie.Iterator(state)
        while it.next():
            if is_leaf(self.__root_path + it.key()):
                yield Path(chr(1) + it.key()), it.data()

        # return {
        #     Path(chr(1) + suffix): self.__trie[(self.__root_path + suffix).key()]
        #     for suffix in self.__trie.suffixes(self.__root_path.key())
        #     if is_leaf(self.__root_path + suffix)
        # }

    def open_leaves(self) -> Generator[Tuple[Path, DerivationTreeNode], None, None]:
        return (
            (self.__to_relative_key(key), self.__trie[key.key()])
            for key in self.__open_leaves
        )

    def get_subtree(self, path: Path) -> "DerivationTree":
        new_root_key = self.__to_absolute_key(path)
        assert new_root_key.key() in self.__trie
        return DerivationTree(
            init_trie=self.__trie,
            root_path=new_root_key,
            open_leaves=self.__open_leaves,
        )

    def add_children(
        self, children: Sequence["DerivationTree"], path: Path = Path()
    ) -> "DerivationTree":
        if len(self.__trie.suffixes(self.__to_absolute_key(path).key())) > 1:
            raise RuntimeError("Cannot add children to an inner node")

        result = self
        for child_idx, child in enumerate(children):
            result = result.replace_path(path + (child_idx,), child)
        return result

    def replace_path(
        self, path: Path, replacement_tree: "DerivationTree"
    ) -> "DerivationTree":
        assert isinstance(path, Path)
        key = self.__to_absolute_key(path)

        if (
            key.key() not in self.__trie
            and len(key) > 1
            and key[:-1].key() not in self.__trie
        ):
            raise RuntimeError(
                f"Cannot replace path {path}, which has no parent in the tree."
            )

        new_open_leaves = {
            self.__to_relative_key(leaf_key)
            for leaf_key in self.__open_leaves
            if leaf_key.key() not in self.__trie.prefixes(key.key())
            and not leaf_key.startswith(key)
        }

        new_trie = datrie.Trie([chr(i) for i in range(32)])
        for k in self.__trie.suffixes(self.__root_path.key()):
            k = chr(1) + k
            if k.startswith(path.key()):
                continue
            new_trie[k] = self.__trie[self.__to_absolute_key(k[1:]).key()]

        new_trie.update(
            {
                (path + repl_tree_suffix).key(): replacement_tree.__trie[
                    (replacement_tree.__root_path + repl_tree_suffix).key()
                ]
                for repl_tree_suffix in replacement_tree.__trie.suffixes(
                    replacement_tree.__root_path.key()
                )
            }
        )

        new_open_leaves.update(
            {
                path + new_leaf_key[len(replacement_tree.__root_path) :]
                for new_leaf_key in replacement_tree.__open_leaves
            }
        )

        for open_leaf in new_open_leaves:
            assert open_leaf.key() in new_trie

        return DerivationTree(
            init_trie=new_trie, root_path=Path(), open_leaves=new_open_leaves
        )

    @staticmethod
    def from_parse_tree(parse_tree: ParseTree) -> "DerivationTree":
        init_map: Dict[Path, str] = {}
        open_leaves: Set[Path] = set()

        stack: List[Tuple[Path, ParseTree]] = [(Path(), parse_tree)]
        while stack:
            path, (node, children) = stack.pop()
            init_map[path] = node
            if children is None:
                open_leaves.add(path)
            for child_idx, child_tree in enumerate(children or []):
                stack.append((path + (child_idx,), child_tree))

        return DerivationTree(init_map=init_map, open_leaves=open_leaves)

    def to_parse_tree(self) -> ParseTree:
        result_stack: List[ParseTree] = []

        for path, node in reversed(list(self.paths())):
            if self.is_open(path):
                result_stack.append((node.value, None))
            elif self.is_leaf(path):
                result_stack.append((node.value, []))
            else:
                children = []
                for _ in range(len(self.children(path))):
                    children.append(result_stack.pop())
                result_stack.append((node.value, children))

        assert len(result_stack) == 1
        return result_stack[0]

    def __getstate__(self) -> bytes:
        return zlib.compress(self.to_json().encode("UTF-8"))

    def __setstate__(self, state: bytes):
        return DerivationTree.from_json(zlib.decompress(state).decode("UTF-8"), self)

    def to_json(self) -> str:
        the_dict = {
            "init_map": dict(self.__trie.items()),
            "open_leaves": tuple(self.__open_leaves),
            "root_path": self.__root_path,
        }

        return json.dumps(the_dict, default=lambda o: o.__dict__)

    @staticmethod
    def from_json(
        json_str: str, tree: Optional["DerivationTree"] = None
    ) -> "DerivationTree":
        the_dict = json.loads(json_str)

        init_map = {
            Path(key): DerivationTreeNode(node["node_id"], node["value"])
            for key, node in the_dict["init_map"].items()
        }

        open_leaves = {Path(leaf["_Path__key"]) for leaf in the_dict["open_leaves"]}
        root_path = Path(the_dict["root_path"]["_Path__key"])

        if tree is not None:
            tree.__open_leaves = open_leaves
            tree.__root_path = root_path
            tree.__trie = DerivationTree.__trie_from_init_map(init_map)
            return tree

        return DerivationTree(
            init_map=init_map,
            open_leaves=open_leaves,
            root_path=root_path,
        )

    def has_unique_ids(self) -> bool:
        nodes = [node for _, node in self.paths()]
        return all(
            nodes[idx_1].node_id != nodes[idx_2].node_id
            for idx_1 in range(len(nodes))
            for idx_2 in range(idx_1 + 1, len(nodes))
        )

    def k_paths(
        self, graph: gg.GrammarGraph, k: int, include_potential_paths: bool = True
    ) -> Set[Tuple[gg.Node, ...]]:
        if (
            include_potential_paths
            and k not in self.__k_paths
            or not include_potential_paths
            and k not in self.__concrete_k_paths
        ):
            paths = graph.k_paths_in_tree(
                self,
                k,
                include_potential_paths=include_potential_paths,
                include_terminals=False,
            )

            if include_potential_paths:
                self.__k_paths[k] = paths
            else:
                self.__concrete_k_paths[k] = paths

        if include_potential_paths:
            return self.__k_paths[k]
        else:
            return self.__concrete_k_paths[k]

    def k_coverage(
        self, graph: gg.GrammarGraph, k: int, include_potential_paths: bool = True
    ) -> float:
        all_paths = graph.k_paths(k, include_terminals=False)
        if not all_paths:
            return 0

        tree_paths = self.k_paths(
            graph, k, include_potential_paths=include_potential_paths
        )

        return len(tree_paths) / len(all_paths)

    def root_nonterminal(self) -> str:
        assert is_nonterminal(self.value())
        return self.value()

    def num_children(self) -> int:
        children = self.children()
        return 0 if children is None else len(children)

    def filter(
        self, f: Callable[["DerivationTreeNode"], bool], enforce_unique: bool = False
    ) -> Dict[Path, "DerivationTree"]:
        result: Dict[Path, "DerivationTree"] = {}

        for path, node in self.paths():
            if f(node):
                result[path] = self.get_subtree(path)

                if enforce_unique and len(result) > 1:
                    raise RuntimeError(
                        f"Found searched-for element more than once in {self}"
                    )

        return result

    def find_node(self, node_or_id: Union[int, "DerivationTree"]) -> Optional[Path]:
        """
        Finds a node by its (assumed unique) ID. Returns the path relative to this node.

        Attention: Might return an empty tuple, which indicates that the searched-for node
        is the root of the tree! Don't use as in `if not find_node(...).`, use
        `if find_node(...) is not None:`.

        :param node_or_id: The node or node ID to search for.
        :return: The path to the node or None.
        """
        if isinstance(node_or_id, DerivationTree):
            node_or_id = node_or_id.node_id()

        try:
            return next(key for key, node in self.paths() if node.node_id == node_or_id)
        except StopIteration:
            return None

    def __traverse_preorder(
        self,
        action: Callable[[Path, DerivationTreeNode], None],
        abort_condition: Callable[
            [Path, DerivationTreeNode], bool
        ] = lambda p, n: False,
    ) -> None:
        # Special iteration for preorder (with normal child order). Since nodes
        # are stored in this order in tries, this is the most efficient kind
        # of traversal.
        for path, node in self.paths():
            if abort_condition(path, node):
                return

            action(path, node)

    def traverse(
        self,
        action: Callable[[Path, DerivationTreeNode], None],
        abort_condition: Callable[
            [Path, DerivationTreeNode], bool
        ] = lambda p, n: False,
        kind: int = TRAVERSE_PREORDER,
        reversed_child_order: bool = False,
    ) -> None:
        if kind == DerivationTree.TRAVERSE_PREORDER and not reversed_child_order:
            self.__traverse_preorder(action, abort_condition)
            return

        assert kind == DerivationTree.TRAVERSE_POSTORDER or reversed_child_order
        if kind == DerivationTree.TRAVERSE_PREORDER:
            reversed_child_order = False

        stack_1: List[Path] = [Path()]
        stack_2: List[Path] = []

        while stack_1:
            path = stack_1.pop()
            node = self.node(path)

            if kind == DerivationTree.TRAVERSE_PREORDER and abort_condition(path, node):
                return

            if kind == DerivationTree.TRAVERSE_POSTORDER:
                stack_2.append(path)

            if kind == DerivationTree.TRAVERSE_PREORDER:
                action(path, node)

            num_children_iterator = range(len(self.children(path) or []))

            for idx in (
                reversed(num_children_iterator)
                if reversed_child_order
                else num_children_iterator
            ):
                new_path = path + (idx,)
                stack_1.append(new_path)

        while kind == DerivationTree.TRAVERSE_POSTORDER and stack_2:
            path = stack_2.pop()
            node = self.node(path)
            if abort_condition(path, node):
                return
            action(path, node)

    def bfs(
        self,
        action: Callable[[Path, DerivationTreeNode], None],
        abort_condition: Callable[
            [Path, DerivationTreeNode], bool
        ] = lambda p, n: False,
    ):
        queue: List[Path] = [Path()]  # FIFO queue
        explored: Set[Path] = {Path()}

        while queue:
            path = queue.pop(0)
            node = self.node(path)

            if abort_condition(path, node):
                return
            action(path, node)

            for child_idx in range(len(self.children(path) or [])):
                child_path = path + (child_idx,)
                if child_path in explored:
                    continue

                explored.add(child_path)
                queue.append(child_path)

    def nonterminals(self) -> Set[str]:
        result: Set[str] = set()

        def add_if_nonterminal(_: Path, node: DerivationTreeNode):
            if is_nonterminal(node.value):
                result.add(node.value)

        self.traverse(action=add_if_nonterminal)

        return result

    def terminals(self) -> Set[str]:
        result: Set[str] = set()

        def add_if_terminal(_: Path, node: DerivationTreeNode):
            if not is_nonterminal(node.value):
                result.add(node.value)

        self.traverse(action=add_if_terminal)

        return result

    def next_path(self, path: Path, skip_children=False) -> Optional[Path]:
        """
        Returns the next path in the tree. Repeated calls result in an iterator
        over the paths in the tree.
        """

        key = self.__to_absolute_key(path)
        suffixes = self.__trie.suffixes(key.key()) if not skip_children else [""]

        if suffixes == [""]:
            while path and key:
                next_sibling_path = key[:-1] + (key[-1] + 1,)
                if next_sibling_path.key() in self.__trie:
                    return path[:-1] + (path[-1] + 1,)

                key = key[:-1]
                path = path[:-1]

            return None

        return path + suffixes[1]

    def new_ids(self) -> "DerivationTree":
        return DerivationTree(
            init_map={
                Path(path_key): DerivationTreeNode(
                    node_id=get_next_id(), value=node.value
                )
                for path_key, node in self.__trie.items()
            },
            open_leaves=self.__open_leaves,
            root_path=self.__root_path,
        )

    def substitute(
        self, subst_map: Dict["DerivationTree", "DerivationTree"]
    ) -> "DerivationTree":
        # Looking up IDs performs much better for big trees, since we do not necessarily
        # have to compute hashes for all nodes. We do not perform "nested" replacements
        # since removing elements in replacements is not intended.
        id_subst_map = {
            tree.node_id(): repl
            for tree, repl in subst_map.items()
            if (
                isinstance(tree, DerivationTree)
                and all(
                    repl.node_id() == tree.node_id()
                    or repl.find_node(tree.node_id()) is None
                    for otree, repl in subst_map.items()
                    if isinstance(otree, DerivationTree)
                )
            )
        }

        result = self
        for tree_id in id_subst_map:
            path = result.find_node(tree_id)
            if path is not None:
                result = result.replace_path(path, id_subst_map[tree_id])

        return result

    def is_prefix(self, other: "DerivationTree") -> bool:
        if len(self) > len(other):
            return False

        for key, node in self.__trie.items():
            key = Path(key)
            value = node.value
            relative_key = self.__to_relative_key(key)
            absolute_other_key = other.__to_absolute_key(relative_key)

            if absolute_other_key.key() not in other.__trie:
                return False

            if value != other.__trie[absolute_other_key.key()].value:
                return False

            if (
                key not in self.__open_leaves
                and len(self.__trie.suffixes(key.key())) == 1
                and len(other.__trie.suffixes(absolute_other_key.key())) > 1
            ):
                return False

        return True

    def is_potential_prefix(self, other: "DerivationTree") -> bool:
        # It's a potential prefix if for all common paths of the two trees, the leaves
        # are equal.
        common_relative_keys = {
            Path(chr(1) + suffix)
            for suffix in self.__trie.suffixes(self.__root_path.key())
        }.intersection(
            {
                Path(chr(1) + suffix)
                for suffix in other.__trie.suffixes(other.__root_path.key())
            }
        )

        for common_relative_key in common_relative_keys:
            if (
                self.__trie[self.__to_absolute_key(common_relative_key).key()].value
                != other.__trie[
                    other.__to_absolute_key(common_relative_key).key()
                ].value
            ):
                return False

        return True

    def __to_relative_key(self, key: Path) -> Path:
        return key[len(self.__root_path) :]

    def __to_absolute_key(self, key: Path) -> Path:
        return self.__root_path + key

    def to_dot(self) -> str:
        dot = Digraph(comment="Derivation Tree")
        dot.attr("node", shape="plain")

        for path, node in self.paths():
            dot.node(
                repr(node.node_id),
                "<"
                + html.escape(node.value)
                + f' <FONT COLOR="gray">({node.node_id})</FONT>>',
            )

            for child_node in (self.children(path) or {}).values():
                dot.edge(repr(node.node_id), repr(child_node.node_id))

        return dot.source

    def __iter__(self) -> Generator[str | List["DerivationTree"] | None, None, None]:
        """
        Allows tuple unpacking: node, children = tree
        This, and getting the value / children via index access, is important for
        backward compatibility to plain `ParseTree` (fuzzingbook) objects.

        :return: An iterator of two elements: The node value and the children's list.
        """
        yield self.value()
        children = self.children()
        yield None if children is None else [
            self.get_subtree(Path((idx,))) for idx in range(len(children))
        ]

    def __getitem__(self, item: int) -> str | Optional[List["DerivationTree"]]:
        """
        Allows accessing the tree's value using index 0 and the children list using
        index 1. For backward compatibility with plain fuzzingbook parse trees.

        :param item: The index of the item to get (0 -> value, 1 -> children)
        :return: The node's value or children list.
        """
        assert isinstance(item, int)
        if not (0 <= item <= 1):
            raise IndexError("Can only access element 0 (node value) or 1 (children)")

        if item == 0:
            return self.value()
        else:
            children = self.children()
            return (
                None
                if children is None
                else [self.get_subtree(Path((idx,))) for idx in range(len(children))]
            )

    def to_string(self, show_open_leaves: bool = False, show_ids: bool = False) -> str:
        result: List[str] = []
        for path, node in self.leaves():
            if self.is_open(path):
                result.append(
                    (f"{node.value} [{node.node_id}]" if show_ids else node.value)
                    if show_open_leaves
                    else ""
                )
            else:
                result.append("" if is_nonterminal(node.value) else node.value)

        return "".join(result)

    def __str__(self) -> str:
        return self.to_string(show_open_leaves=True)

    def __repr__(self) -> str:
        return (
            "DerivationTree(init_map="
            + f"{repr({Path(path): node for path, node in self.__trie.items()})}, "
            + f"open_leaves={repr(self.__open_leaves)}, "
            + f"root_path={repr(self.__root_path)})"
        )

    def __len__(self):
        return len(self.__trie.suffixes(self.__root_path.key()))

    def __hash__(self):
        return hash((tuple(self.paths()), tuple(self.open_leaves())))

    def structural_hash(self):
        return hash(
            (
                tuple([(path, node.value) for path, node in self.paths()]),
                tuple(
                    [(path, node.value) for path, node in self.open_leaves()]
                ),
            )
        )

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, DerivationTree)
            and iterators_equal(self.open_leaves(), other.open_leaves())
            and iterators_equal(self.paths(), other.paths())
        )

    def structurally_equal(self, other) -> bool:
        return (
            isinstance(other, DerivationTree)
            and iterators_equal(self.open_leaves(), other.open_leaves())
            and all(
                (p1, v1.value) == (p2, v2.value)
                for (p1, v1), (p2, v2) in itertools.zip_longest(
                    self.paths(), other.paths()
                )
            )
        )


RE_NONTERMINAL = re.compile(r"(<[^<> ]*>)")


@lru_cache(maxsize=None)
def is_nonterminal(s):
    return RE_NONTERMINAL.match(s)


def iterators_equal(iterator_1, iterator_2) -> bool:
    return all(
        elem_1 == elem_2
        for elem_1, elem_2 in itertools.zip_longest(iterator_1, iterator_2)
    )
