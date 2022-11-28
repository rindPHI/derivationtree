import random
import re
import unittest
from typing import Dict

from dtree import (
    ParseTree,
    Path,
    DerivationTree,
    DerivationTreeNode,
    trie_key_to_path,
    path_to_trie_key,
)


class TestDerivationTree(unittest.TestCase):
    def test_paths_preserved(self):
        parse_tree = ("A", [("B", []), ("C", [("D", None), ("E", [])])])
        paths = dict(parse_tree_paths(parse_tree))
        dtree = DerivationTree(init_map=paths)

        self.assertEqual(paths, dtree_paths_to_parse_tree_paths(dtree.paths()))

    def test_get_subtree(self):
        parse_tree = ("A", [("B", []), ("C", [("D", None), ("E", [])])])
        paths = dict(parse_tree_paths(parse_tree))
        dtree = DerivationTree(init_map=paths)

        sub_parse_tree_1 = get_subtree(parse_tree, (1,))
        sub_dtree_1 = dtree.get_subtree((1,))

        self.assertEqual(
            parse_tree_paths(sub_parse_tree_1),
            dtree_paths_to_parse_tree_paths(sub_dtree_1.paths()),
        )

        sub_parse_tree_2 = get_subtree(sub_parse_tree_1, (0,))
        sub_dtree_2 = sub_dtree_1.get_subtree((0,))

        self.assertEqual(
            parse_tree_paths(sub_parse_tree_2),
            dtree_paths_to_parse_tree_paths(sub_dtree_2.paths()),
        )

    def test_leaves(self):
        parse_tree = ("A", [("B", []), ("C", [("D", None), ("E", [])])])
        paths = dict(parse_tree_paths(parse_tree))
        dtree = DerivationTree(init_map=paths)

        self.assertEqual(
            tree_leaves(parse_tree), dtree_paths_to_parse_tree_paths(dtree.leaves())
        )

        self.assertEqual(
            tree_leaves(get_subtree(parse_tree, (0,))),
            dtree_paths_to_parse_tree_paths(dtree.get_subtree((0,)).leaves()),
        )

        self.assertEqual(
            tree_leaves(get_subtree(parse_tree, (1,))),
            dtree_paths_to_parse_tree_paths(dtree.get_subtree((1,)).leaves()),
        )

    def test_open_leaves(self):
        parse_tree = ("A", [("B", []), ("C", [("D", None), ("E", [])])])
        paths = dict(parse_tree_paths(parse_tree))
        dtree = DerivationTree(init_map=paths, open_leaves={(1, 0)})

        self.assertEqual(
            open_leaves(parse_tree),
            dtree_paths_to_parse_tree_paths(dtree.open_leaves()),
        )

        self.assertEqual(
            open_leaves(get_subtree(parse_tree, (0,))),
            dtree_paths_to_parse_tree_paths(dtree.get_subtree((0,)).open_leaves()),
        )

        self.assertEqual(
            open_leaves(get_subtree(parse_tree, (1,))),
            dtree_paths_to_parse_tree_paths(dtree.get_subtree((1,)).open_leaves()),
        )

        self.assertEqual(
            open_leaves(get_subtree(parse_tree, (1, 0))),
            dtree_paths_to_parse_tree_paths(dtree.get_subtree((1, 0)).open_leaves()),
        )

    def test_from_parse_tree(self):
        parse_tree = ("A", [("B", []), ("C", [("D", None), ("E", [])])])
        dtree = DerivationTree.from_parse_tree(parse_tree)

        self.assertEqual(
            parse_tree_paths(parse_tree), dtree_paths_to_parse_tree_paths(dtree.paths())
        )

        self.assertEqual(
            tree_leaves(parse_tree), dtree_paths_to_parse_tree_paths(dtree.leaves())
        )

        self.assertEqual(
            open_leaves(parse_tree),
            dtree_paths_to_parse_tree_paths(dtree.open_leaves()),
        )

    def test_get_node(self):
        parse_tree = ("A", [("B", []), ("C", [("D", None), ("E", [])])])
        dtree = DerivationTree.from_parse_tree(parse_tree)

        for path, value in parse_tree_paths(parse_tree).items():
            self.assertEqual(value, dtree.node(path).value)

    def test_children(self):
        parse_tree = ("A", [("B", []), ("C", [("D", None), ("E", [])])])
        dtree = DerivationTree.from_parse_tree(parse_tree)

        self.assertEqual(["B", "C"], [node.value for node in dtree.children()])
        self.assertEqual(["D", "E"], [node.value for node in dtree.children((1,))])
        self.assertEqual(
            ["D", "E"], [node.value for node in dtree.get_subtree((1,)).children()]
        )

        self.assertEqual([], dtree.children((0,)))
        self.assertEqual(None, dtree.children((1, 0)))

        self.assertEqual([], dtree.get_subtree((0,)).children())
        self.assertEqual(None, dtree.get_subtree((1, 0)).children())

    def test_is_leaf(self):
        parse_tree = ("A", [("B", []), ("C", [("D", None), ("E", [])])])
        dtree = DerivationTree.from_parse_tree(parse_tree)

        all_paths = set(parse_tree_paths(parse_tree))
        leaf_paths = set(tree_leaves(parse_tree))

        for path in all_paths:
            self.assertEqual(path in leaf_paths, dtree.is_leaf(path))

        sub_parse_tree = get_subtree(parse_tree, (1,))
        sub_dtree = dtree.get_subtree((1,))

        all_paths = set(parse_tree_paths(sub_parse_tree))
        leaf_paths = set(tree_leaves(sub_parse_tree))

        for path in all_paths:
            self.assertEqual(path in leaf_paths, sub_dtree.is_leaf(path))

    def test_is_open(self):
        parse_tree = ("A", [("B", []), ("C", [("D", None), ("E", [])])])
        dtree = DerivationTree.from_parse_tree(parse_tree)

        all_paths = set(parse_tree_paths(parse_tree))
        open_paths = set(open_leaves(parse_tree))

        for path in all_paths:
            self.assertEqual(path in open_paths, dtree.is_open(path))

        sub_parse_tree = get_subtree(parse_tree, (1,))
        sub_dtree = dtree.get_subtree((1,))

        all_paths = set(parse_tree_paths(sub_parse_tree))
        open_paths = set(open_leaves(sub_parse_tree))

        for path in all_paths:
            self.assertEqual(path in open_paths, sub_dtree.is_open(path))

    def test_to_parse_tree(self):
        parse_tree = ("A", [("B", []), ("C", [("D", None), ("E", [])])])
        self.assertEqual(
            parse_tree, DerivationTree.from_parse_tree(parse_tree).to_parse_tree()
        )

    def test_replace_path(self):
        parse_tree = ("A", [("B", []), ("C", [("D", None), ("E", [])])])
        replacement_tree = ("F", [("G", []), ("H", [("I", []), ("J", None)])])

        dtree = DerivationTree.from_parse_tree(parse_tree)
        replacement_dtree = DerivationTree.from_parse_tree(replacement_tree)

        expected = replace_path(parse_tree, (1,), replacement_tree)
        result = dtree.replace_path((1,), replacement_dtree)

        self.assertEqual(expected, result.to_parse_tree())
        self.assertEqual(replacement_dtree, result.get_subtree((1,)))

        sub_tree = get_subtree(parse_tree, (1,))
        sub_dtree = dtree.get_subtree((1,))
        sub_replacement_tree = get_subtree(replacement_tree, (1, 0))
        sub_replacement_dtree = replacement_dtree.get_subtree((1, 0))

        self.assertEqual(sub_tree, sub_dtree.to_parse_tree())
        self.assertEqual(sub_replacement_tree, sub_replacement_dtree.to_parse_tree())

        expected = replace_path(sub_tree, (1,), sub_replacement_tree)
        result = sub_dtree.replace_path((1,), sub_replacement_dtree)

        self.assertEqual(expected, result.to_parse_tree())

    def test_repr(self):
        parse_tree = ("A", [("B", []), ("C", [("D", None), ("E", [])])])
        dtree = DerivationTree.from_parse_tree(parse_tree)
        self.assertEqual(dtree, eval(repr(dtree)))

    def test_str(self):
        parse_tree = ("A", [("B", []), ("C", [("D", None), ("E", [])])])
        dtree = DerivationTree.from_parse_tree(parse_tree)

        expected = (
            "(DerivationTreeNode(node_id=XXX, value='A'), ["
            + "(DerivationTreeNode(node_id=XXX, value='B'), []), "
            + "(DerivationTreeNode(node_id=XXX, value='C'), ["
            + "(DerivationTreeNode(node_id=XXX, value='D'), None), "
            + "(DerivationTreeNode(node_id=XXX, value='E'), [])])])"
        )

        regex = re.compile(r"node_id=" + "[0-9]+")
        self.assertEqual(expected, regex.sub("node_id=XXX", str(dtree)))

    def test_tree_is_open(self):
        parse_tree = ("A", [("B", []), ("C", [("D", None), ("E", [])])])
        dtree = DerivationTree.from_parse_tree(parse_tree)

        self.assertTrue(dtree.tree_is_open())
        self.assertTrue(dtree.get_subtree((1,)).tree_is_open())
        self.assertTrue(dtree.get_subtree((1, 0)).tree_is_open())
        self.assertFalse(dtree.get_subtree((0,)).tree_is_open())
        self.assertFalse(dtree.get_subtree((1, 1)).tree_is_open())

    def test_unpack(self):
        parse_tree = ("A", [("B", []), ("C", [("D", None), ("E", [])])])
        dtree = DerivationTree.from_parse_tree(parse_tree)

        node, children = dtree
        self.assertEqual("A", node)
        self.assertEqual(node, dtree.value())
        self.assertEqual(2, len(children))
        self.assertEqual(["B", "C"], [child.value() for child in children])

        node, children = dtree.get_subtree((1, 0))
        self.assertEqual("D", node)
        self.assertEqual(None, children)

        node, children = dtree.get_subtree((0,))
        self.assertEqual("B", node)
        self.assertEqual([], children)

    def test_tuple_access(self):
        parse_tree = ("A", [("B", []), ("C", [("D", None), ("E", [])])])
        dtree = DerivationTree.from_parse_tree(parse_tree)

        node, children = dtree[0], dtree[1]
        self.assertEqual("A", node)
        self.assertEqual(node, dtree.value())
        self.assertEqual(2, len(children))
        self.assertEqual(["B", "C"], [child.value() for child in children])

        subtree = dtree.get_subtree((1, 0))
        node, children = subtree[0], subtree[1]
        self.assertEqual("D", node)
        self.assertEqual(None, children)

        subtree = dtree.get_subtree((0,))
        node, children = subtree[0], subtree[1]
        self.assertEqual("B", node)
        self.assertEqual([], children)

        try:
            _ = dtree[2]
            self.fail("IndexError Expected")
        except Exception as ierr:
            self.assertTrue(isinstance(ierr, IndexError))

    def test_hash(self):
        dtree_1 = DerivationTree(
            init_map={(): DerivationTreeNode(1, "A"), (0,): DerivationTreeNode(2, "B")}
        )

        dtree_2 = DerivationTree(
            init_map={(): DerivationTreeNode(1, "A"), (0,): DerivationTreeNode(2, "B")}
        )

        self.assertEqual(hash(dtree_1), hash(dtree_2))

        dtree_3 = DerivationTree(
            init_map={(): DerivationTreeNode(1, "A"), (0,): DerivationTreeNode(3, "B")}
        )

        self.assertNotEqual(hash(dtree_1), hash(dtree_3))

        dtree_4 = DerivationTree(
            init_map={(): DerivationTreeNode(1, "A"), (0,): DerivationTreeNode(2, "B")},
            open_leaves={(0,)},
        )

        self.assertNotEqual(hash(dtree_1), hash(dtree_4))

    def test_structural_hash(self):
        dtree_1 = DerivationTree(
            init_map={(): DerivationTreeNode(1, "A"), (0,): DerivationTreeNode(2, "B")}
        )

        dtree_2 = DerivationTree(
            init_map={(): DerivationTreeNode(1, "A"), (0,): DerivationTreeNode(2, "B")}
        )

        self.assertEqual(dtree_1.structural_hash(), dtree_2.structural_hash())

        dtree_3 = DerivationTree(
            init_map={(): DerivationTreeNode(1, "A"), (0,): DerivationTreeNode(3, "B")}
        )

        self.assertEqual(dtree_1.structural_hash(), dtree_3.structural_hash())

        dtree_4 = DerivationTree(
            init_map={(): DerivationTreeNode(1, "A"), (0,): DerivationTreeNode(2, "B")},
            open_leaves={(0,)},
        )

        self.assertNotEqual(dtree_1.structural_hash(), dtree_4.structural_hash())

    def test_eq(self):
        dtree_1 = DerivationTree(
            init_map={(): DerivationTreeNode(1, "A"), (0,): DerivationTreeNode(2, "B")}
        )

        dtree_2 = DerivationTree(
            init_map={(): DerivationTreeNode(1, "A"), (0,): DerivationTreeNode(2, "B")}
        )

        self.assertEqual(dtree_1, dtree_2)

        dtree_3 = DerivationTree(
            init_map={(): DerivationTreeNode(1, "A"), (0,): DerivationTreeNode(3, "B")}
        )

        self.assertNotEqual(dtree_1, dtree_3)

        dtree_4 = DerivationTree(
            init_map={(): DerivationTreeNode(1, "A"), (0,): DerivationTreeNode(2, "B")},
            open_leaves={(0,)},
        )

        self.assertNotEqual(dtree_1, dtree_4)

    def test_structurally_equal(self):
        dtree_1 = DerivationTree(
            init_map={(): DerivationTreeNode(1, "A"), (0,): DerivationTreeNode(2, "B")}
        )

        dtree_2 = DerivationTree(
            init_map={(): DerivationTreeNode(1, "A"), (0,): DerivationTreeNode(2, "B")}
        )

        self.assertTrue(dtree_1.structurally_equal(dtree_2))

        dtree_3 = DerivationTree(
            init_map={(): DerivationTreeNode(1, "A"), (0,): DerivationTreeNode(3, "B")}
        )

        self.assertTrue(dtree_1.structurally_equal(dtree_3))

        dtree_4 = DerivationTree(
            init_map={(): DerivationTreeNode(1, "A"), (0,): DerivationTreeNode(2, "B")},
            open_leaves={(0,)},
        )

        self.assertFalse(dtree_1.structurally_equal(dtree_4))

    def test_trie_key_to_path(self):
        for _ in range(30):
            random_path = tuple(
                [random.randint(0, 30) for _ in range(random.randint(1, 20))]
            )
            self.assertEqual(
                random_path, trie_key_to_path(path_to_trie_key(random_path))
            )

        try:
            trie_key_to_path("Hallo!")
        except Exception as err:
            self.assertIsInstance(err, RuntimeError)
            self.assertIn("Invalid trie key", str(err))
            self.assertIn("should start with 1", str(err))

    def test_path_index_out_of_range(self):
        tree = DerivationTree(init_map={(): "A", (29,): "B"})
        self.assertEqual(2, len(tree.paths()))

        # For path indices larger than 29, the respective subtrees vanish.
        # That's not really a feature; this test case means to document that
        # particularity. If we want to support grammars with more than 30 (incl.
        # 0) direct children of a node, the vocabulary of the datrie structure
        # must be increased.
        tree = DerivationTree(init_map={(): "A", (30,): "B"})
        self.assertEqual(1, len(tree.paths()))
        self.assertEqual(("A", []), tree.to_parse_tree())


def dtree_paths_to_parse_tree_paths(
    dtree_paths: Dict[Path, DerivationTreeNode]
) -> Dict[Path, str]:
    return {path: node.value for path, node in dtree_paths.items()}


def parse_tree_paths(parse_tree: ParseTree) -> Dict[Path, str]:
    node, children = parse_tree
    if not children:
        return {(): node}

    return {(): node} | {
        (child_idx,) + child_path: child_node
        for child_idx in range(len(children))
        for child_path, child_node in parse_tree_paths(children[child_idx]).items()
    }


def get_subtree(tree: ParseTree, path: Path) -> ParseTree:
    """Access a subtree based on `path` (a list of children numbers)"""
    curr_node = tree
    while path:
        curr_node = curr_node[1][path[0]]
        path = path[1:]

    return curr_node


def open_leaves(tree: ParseTree) -> Dict[Path, str]:
    return {
        path: get_subtree(tree, path)[0]
        for path in parse_tree_paths(tree)
        if get_subtree(tree, path)[1] is None
    }


def tree_leaves(tree: ParseTree) -> Dict[Path, str]:
    return {
        path: get_subtree(tree, path)[0]
        for path in parse_tree_paths(tree)
        if not get_subtree(tree, path)[1]
    }


def replace_path(tree: ParseTree, path: Path, new_subtree: ParseTree) -> ParseTree:
    if not path:
        return new_subtree

    node, children = tree
    return (
        node,
        children[: path[0]]
        + [replace_path(children[path[0]], path[1:], new_subtree)]
        + children[path[0] + 1 :],
    )


if __name__ == "__main__":
    unittest.main()
