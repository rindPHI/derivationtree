import copy
import re
import zlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple, List, Optional, Dict, Iterable, Set, Generator, Callable

import datrie

from grammar_graph import gg

Path = Tuple[int, ...]
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


class DerivationTree:
    TRAVERSE_PREORDER = 0
    TRAVERSE_POSTORDER = 1

    def __init__(
        self,
        init_map: Optional[Dict[Path | str, DerivationTreeNode | str]] = None,
        init_trie: Optional[datrie.Trie] = None,
        open_leaves: Optional[Iterable[Path | str]] = None,
        root_path: Optional[Path | str] = None,
    ):
        assert init_map or init_trie

        self.__trie: datrie.Trie = init_trie or datrie.Trie([chr(i) for i in range(32)])

        for path, node_or_value in (init_map or {}).items():
            self.__trie[path_to_trie_key(path) if isinstance(path, tuple) else path] = (
                node_or_value
                if isinstance(node_or_value, DerivationTreeNode)
                else DerivationTreeNode(get_next_id(), node_or_value)
            )

        self.__root_path = (
            path_to_trie_key(root_path)
            if isinstance(root_path, tuple)
            else (root_path or chr(1))
        )

        self.__open_leaves: Set[str] = {
            path_to_trie_key(path) if isinstance(path, tuple) else path
            for path in open_leaves or []
            if (path_to_trie_key(path) if isinstance(path, tuple) else path).startswith(
                self.__root_path
            )
        }

    def node(self, path: Path = ()) -> DerivationTreeNode:
        return self.__trie[self.__root_path + path_to_trie_key(path)[1:]]

    def value(self, path: Path = ()) -> str:
        return self.__trie[self.__root_path + path_to_trie_key(path)[1:]].value

    def children(self, path: Path = ()) -> Optional[List[DerivationTreeNode]]:
        if self.is_open(path):
            return None

        if self.is_leaf(path):
            return []

        result: List[DerivationTreeNode] = []

        i = 0
        while True:
            try:
                result.append(
                    self.__trie[
                        self.__root_path + path_to_trie_key(path)[1:] + chr(i + 2)
                    ]
                )
                i += 1
            except KeyError:
                break

        return result

    def is_leaf(self, path: Path = ()) -> bool:
        return (
            len(self.__trie.suffixes(self.__root_path + path_to_trie_key(path)[1:]))
            == 1
        )

    def is_open(self, path: Path = ()) -> bool:
        return self.__root_path + path_to_trie_key(path)[1:] in self.__open_leaves

    def is_complete(self, path: Path = ()) -> bool:
        return not self.is_open(path)

    def tree_is_open(self):
        return bool(self.__open_leaves)

    def is_valid_path(self, path: Path) -> bool:
        return self.__root_path + path_to_trie_key(path)[1:] in self.__trie

    def paths(self) -> Dict[Path, DerivationTreeNode]:
        return {
            trie_key_to_path(chr(1) + suffix): self.__trie[self.__root_path + suffix]
            for suffix in self.__trie.suffixes(self.__root_path)
        }

    @lru_cache
    def leaves(self) -> Dict[Path, DerivationTreeNode]:
        return {
            trie_key_to_path(chr(1) + suffix): self.__trie[self.__root_path + suffix]
            for suffix in self.__trie.suffixes(self.__root_path)
            if len(self.__trie.suffixes(self.__root_path + suffix)) == 1
        }

    def open_leaves(self) -> Dict[Path, DerivationTreeNode]:
        return {
            trie_key_to_path(chr(1) + path[len(self.__root_path) :]): self.__trie[path]
            for path in self.__open_leaves
        }

    def get_subtree(self, path: Path) -> "DerivationTree":
        assert path_to_trie_key(path) in self.__trie
        return DerivationTree(
            init_trie=self.__trie,
            root_path=self.__root_path + path_to_trie_key(path)[1:],
            open_leaves=self.__open_leaves,
        )

    def replace_path(
        self, path: Path, replacement_tree: "DerivationTree"
    ) -> "DerivationTree":
        new_trie = copy.deepcopy(self.__trie)
        new_open_leaves = copy.deepcopy(self.__open_leaves)

        key = self.__root_path + path_to_trie_key(path)[1:]

        for old_suffix in self.__trie.suffixes(key):
            old_leaf_key = key + old_suffix
            del new_trie[old_leaf_key]
            if old_leaf_key in new_open_leaves:
                new_open_leaves.remove(old_leaf_key)

        new_trie.update(
            {
                key
                + repl_tree_suffix: replacement_tree.__trie[
                    replacement_tree.__root_path + repl_tree_suffix
                ]
                for repl_tree_suffix in replacement_tree.__trie.suffixes(
                    replacement_tree.__root_path
                )
            }
        )

        new_open_leaves.update(
            {
                self.__root_path
                + key[1:]
                + new_leaf_key[len(replacement_tree.__root_path) :]
                for new_leaf_key in replacement_tree.__open_leaves
            }
        )

        return DerivationTree(
            init_trie=new_trie, root_path=self.__root_path, open_leaves=new_open_leaves
        )

    @staticmethod
    def from_parse_tree(parse_tree: ParseTree) -> "DerivationTree":
        init_map: Dict[Path, str] = {}
        open_leaves: Set[Path] = set()

        stack: List[Tuple[Path, ParseTree]] = [((), parse_tree)]
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

        for path in reversed(self.paths()):
            if self.is_open(path):
                result_stack.append((self.node(path).value, None))
            elif self.is_leaf(path):
                result_stack.append((self.node(path).value, []))
            else:
                children = []
                for _ in range(len(self.children(path))):
                    children.append(result_stack.pop())
                result_stack.append((self.node(path).value, children))

        assert len(result_stack) == 1
        return result_stack[0]

    def __getstate__(self) -> bytes:
        return zlib.compress(self.to_json().encode("UTF-8"))

    def __setstate__(self, state: bytes):
        return DerivationTree.from_json(zlib.decompress(state).decode("UTF-8"), self)

    def to_json(self) -> str:
        # TODO
        raise NotImplementedError

    @staticmethod
    def from_json(
        json_str: str, tree: Optional["DerivationTree"] = None
    ) -> "DerivationTree":
        # TODO
        raise NotImplementedError

    def has_unique_ids(self) -> bool:
        # TODO
        raise NotImplementedError

    def k_coverage(
        self, graph: gg.GrammarGraph, k: int, include_potential_paths: bool = True
    ) -> float:
        # TODO
        raise NotImplementedError

    def k_paths(
        self, graph: gg.GrammarGraph, k: int, include_potential_paths: bool = True
    ) -> Set[Tuple[gg.Node, ...]]:
        # TODO
        raise NotImplementedError

    def root_nonterminal(self) -> str:
        assert is_nonterminal(self.value())
        return self.value()

    def num_children(self) -> int:
        children = self.children()
        return 0 if children is None else len(children)

    def filter(
        self, f: Callable[["DerivationTree"], bool], enforce_unique: bool = False
    ) -> List[Tuple[Path, "DerivationTree"]]:
        # TODO
        raise NotImplementedError

    def find_node(self, node_or_id: "DerivationTree" | int) -> Optional[Path]:
        # TODO
        raise NotImplementedError

    def traverse(
        self,
        action: Callable[[Path, "DerivationTree"], None],
        abort_condition: Callable[[Path, "DerivationTree"], bool] = lambda p, n: False,
        kind: int = TRAVERSE_PREORDER,
        reverse: bool = False,
    ) -> None:
        # TODO
        raise NotImplementedError

    def bfs(
        self,
        action: Callable[[Path, "DerivationTree"], None],
        abort_condition: Callable[[Path, "DerivationTree"], bool] = lambda p, n: False,
    ):
        # TODO
        raise NotImplementedError

    def nonterminals(self) -> Set[str]:
        # TODO
        raise NotImplementedError

    def terminals(self) -> Set[str]:
        # TODO
        raise NotImplementedError

    def next_path(self, path: Path, skip_children=False) -> Optional[Path]:
        """
        Returns the next path in the tree. Repeated calls result in an iterator over the paths in the tree.
        """
        # TODO
        raise NotImplementedError

    def new_ids(self) -> "DerivationTree":
        # TODO
        raise NotImplementedError

    def substitute(
        self, subst_map: Dict["DerivationTree", "DerivationTree"]
    ) -> "DerivationTree":
        # TODO
        raise NotImplementedError

    def is_prefix(self, other: "DerivationTree") -> bool:
        # TODO
        raise NotImplementedError

    def is_potential_prefix(self, other: "DerivationTree") -> bool:
        # It's a potential prefix if for all common paths of the two trees, the leaves
        # are equal.
        # TODO
        raise NotImplementedError

    def to_string(self, show_open_leaves: bool = False, show_ids: bool = False) -> str:
        # TODO
        raise NotImplementedError

    def to_dot(self) -> str:
        # TODO
        raise NotImplementedError

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
            self.get_subtree((idx,)) for idx in range(len(children))
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
                else [self.get_subtree((idx,)) for idx in range(len(children))]
            )

    def __str__(self) -> str:
        result_stack: List[str] = []

        for path in reversed(self.paths()):
            if self.is_open(path):
                result_stack.append(f"({self.node(path)}, None)")
            elif self.is_leaf(path):
                result_stack.append(f"({self.node(path)}, [])")
            else:
                children = []
                for _ in range(len(self.children(path))):
                    children.append(result_stack.pop())
                result_stack.append(f"({self.node(path)}, [{', '.join(children)}])")

        assert len(result_stack) == 1
        return result_stack[0]

    def __repr__(self) -> str:
        return (
            "DerivationTree(init_map="
            + f"{repr({path: node for path, node in self.__trie.items()})}, "
            + f"open_leaves={repr(self.__open_leaves)}, "
            + f"root_path={repr(self.__root_path)})"
        )

    def __len__(self):
        return len(self.__trie.suffixes(self.__root_path))

    def __hash__(self):
        return hash((tuple(self.paths().items()), tuple(self.open_leaves().items())))

    def structural_hash(self):
        return hash(
            (
                tuple({path: node.value for path, node in self.paths().items()}),
                tuple({path: node.value for path, node in self.open_leaves().items()}),
            )
        )

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, DerivationTree)
            and len(self) == len(other)
            and self.paths() == other.paths()
            and self.open_leaves() == other.open_leaves()
        )

    def structurally_equal(self, other) -> bool:
        return (
            isinstance(other, DerivationTree)
            and len(self) == len(other)
            and self.paths().keys() == other.paths().keys()
            and self.open_leaves().keys() == other.open_leaves().keys()
            and all(self.value(path) == other.value(path) for path in self.paths())
        )


def path_to_trie_key(path: Path) -> str:
    # 0-bytes are ignored by the trie ==> +1
    # To represent the empty part, reserve chr(1) ==> +2
    if not path:
        return chr(1)

    return chr(1) + "".join([chr(i + 2) for i in path])


def trie_key_to_path(key: str) -> Path:
    if not key or key[0] != chr(1):
        raise RuntimeError(
            f"Invalid trie key '{key}' ({[ord(c) for c in key]}), should start with 1"
        )

    if key == chr(1):
        return ()

    return tuple([ord(c) - 2 for c in key if ord(c) != 1])


RE_NONTERMINAL = re.compile(r"(<[^<> ]*>)")


@lru_cache(maxsize=None)
def is_nonterminal(s):
    return RE_NONTERMINAL.match(s)
