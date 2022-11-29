import random
import re
import string
import unittest
from typing import Dict, List

from grammar_graph import gg

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

        self.assertEqual("BDE", str(dtree))
        self.assertEqual("BE", dtree.to_string(show_open_leaves=False))
        self.assertEqual(
            f"BD [{dtree.node_id((1, 0))}]E",
            dtree.to_string(show_open_leaves=True, show_ids=True),
        )

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

    def test_to_from_json(self):
        dtree = DerivationTree(
            init_map={
                (): DerivationTreeNode(1, "A"),
                (0,): DerivationTreeNode(2, "B"),
                (1,): DerivationTreeNode(3, "C"),
                (0, 0): DerivationTreeNode(4, "D"),
                (0, 0, 0): DerivationTreeNode(5, "E"),
                (0, 0, 1): DerivationTreeNode(6, "F"),
            },
            open_leaves={(1,), (0, 0, 0)},
        )

        self.assertEqual(dtree, DerivationTree.from_json(dtree.to_json()))

    def test_pickle_unpickle(self):
        dtree = DerivationTree(
            init_map={
                (): DerivationTreeNode(1, "A"),
                (0,): DerivationTreeNode(2, "B"),
                (1,): DerivationTreeNode(3, "C"),
                (0, 0): DerivationTreeNode(4, "D"),
                (0, 0, 0): DerivationTreeNode(5, "E"),
                (0, 0, 1): DerivationTreeNode(6, "F"),
            },
            open_leaves={(1,), (0, 0, 0)},
        )

    def test_unique_ids(self):
        dtree = DerivationTree(
            init_map={
                (): DerivationTreeNode(1, "A"),
                (0,): DerivationTreeNode(2, "B"),
                (1,): DerivationTreeNode(3, "C"),
                (0, 0): DerivationTreeNode(4, "D"),
                (0, 0, 0): DerivationTreeNode(5, "E"),
                (0, 0, 1): DerivationTreeNode(6, "F"),
            },
            open_leaves={(1,), (0, 0, 0)},
        )

        self.assertTrue(dtree.has_unique_ids())

        self.assertFalse(
            dtree.replace_path(
                (0, 0), DerivationTree(init_map={(): DerivationTreeNode(3, "D")})
            ).has_unique_ids()
        )

    def test_k_paths(self):
        grammar = {
            "<start>": ["<stmt>"],
            "<stmt>": ["<assgn> ; <stmt>", "<assgn>"],
            "<assgn>": ["<var> := <rhs>"],
            "<rhs>": ["<var>", "<digit>"],
            "<var>": list(string.ascii_lowercase),
            "<digit>": list(string.digits),
        }

        graph = gg.GrammarGraph.from_grammar(grammar)

        dtree = DerivationTree.from_parse_tree(
            (
                "<start>",
                [
                    (
                        "<stmt>",
                        [
                            (
                                "<assgn>",
                                [
                                    ("<var>", None),
                                    (" := ", []),
                                    ("<rhs>", [("<digit>", [("1", [])])]),
                                ],
                            ),
                            (" ; ", []),
                            ("<stmt>", [("<assgn>", None)]),
                        ],
                    )
                ],
            )
        )

        self.assertTrue(graph.tree_is_valid(dtree))

        concrete_three_paths = {
            gg.path_to_string(path, include_choice_node=False)
            for path in dtree.k_paths(graph, 3, include_potential_paths=False)
        }

        self.assertEqual(
            {
                "<start> <stmt> <assgn>",
                "<start> <stmt> <stmt>",
                "<stmt> <assgn> <var>",
                "<stmt> <assgn> <rhs>",
                "<stmt> <stmt> <assgn>",
                "<assgn> <rhs> <digit>",
            },
            concrete_three_paths,
        )

        potential_three_paths = {
            gg.path_to_string(path, include_choice_node=False)
            for path in dtree.k_paths(graph, 3, include_potential_paths=True)
        }

        self.assertTrue(not concrete_three_paths.difference(potential_three_paths))
        potential_only_paths = potential_three_paths.difference(concrete_three_paths)
        self.assertEqual({"<assgn> <rhs> <var>"}, potential_only_paths)

    def test_k_path_coverage(self):
        grammar = {
            "<start>": ["<stmt>"],
            "<stmt>": ["<assgn> ; <stmt>", "<assgn>"],
            "<assgn>": ["<var> := <rhs>"],
            "<rhs>": ["<var>", "<digit>"],
            "<var>": list(string.ascii_lowercase),
            "<digit>": list(string.digits),
        }

        graph = gg.GrammarGraph.from_grammar(grammar)

        dtree = DerivationTree.from_parse_tree(
            (
                "<start>",
                [
                    (
                        "<stmt>",
                        [
                            (
                                "<assgn>",
                                [
                                    ("<var>", None),
                                    (" := ", []),
                                    ("<rhs>", [("<digit>", [("1", [])])]),
                                ],
                            ),
                            (" ; ", []),
                            ("<stmt>", [("<assgn>", None)]),
                        ],
                    )
                ],
            )
        )

        # Missing paths:
        # - <start> <start>-choice-1 <stmt> <stmt>-choice-2 <assgn>
        # - <stmt> <stmt>-choice-1 <stmt> <stmt>-choice-1 <assgn>
        # - <stmt> <stmt>-choice-1 <stmt> <stmt>-choice-1 <stmt>
        self.assertEqual(0.75, dtree.k_coverage(graph, 3, include_potential_paths=True))

        # Additional missing paths when excluding potential ones:
        # - <stmt> <stmt>-choice-1 <stmt> <stmt>-choice-1 <assgn>
        # - <stmt> <stmt>-choice-2 <assgn> <assgn>-choice-1 <rhs>
        # - <start> <start>-choice-1 <stmt> <stmt>-choice-2 <assgn>
        # - <stmt> <stmt>-choice-1 <stmt> <stmt>-choice-1 <assgn>
        # - <assgn> <assgn>-choice-1 <rhs> <rhs>-choice-1 <var>
        # - <stmt> <stmt>-choice-2 <assgn> <assgn>-choice-1 <var>
        self.assertEqual(0.5, dtree.k_coverage(graph, 3, include_potential_paths=False))

    def test_filter(self):
        dtree = DerivationTree(
            init_map={
                (): DerivationTreeNode(1, "A"),
                (0,): DerivationTreeNode(2, "B"),
                (1,): DerivationTreeNode(3, "C"),
                (0, 0): DerivationTreeNode(4, "A"),
                (0, 0, 0): DerivationTreeNode(5, "B"),
                (0, 0, 1): DerivationTreeNode(6, "D"),
            },
            open_leaves={(1,), (0, 0, 0)},
        )

        self.assertEqual(2, len(dtree.filter(lambda node: node.value == "B")))
        self.assertEqual(1, len(dtree.filter(lambda node: node.value == "D")))
        self.assertEqual(
            1, len(dtree.filter(lambda node: node.value == "D", enforce_unique=True))
        )

        try:
            dtree.filter(lambda node: node.value == "B", enforce_unique=True)
            self.fail("Error excpected")
        except Exception as exc:
            self.assertIsInstance(exc, RuntimeError)
            self.assertIn("more than once", str(exc))

    def test_find_start(self):
        tree = DerivationTree(init_map={(): DerivationTreeNode(1, "<start>")})
        self.assertEqual((), tree.find_node(1))

    def test_find(self):
        dtree = DerivationTree(
            init_map={
                (): DerivationTreeNode(1, "A"),
                (0,): DerivationTreeNode(2, "B"),
                (1,): DerivationTreeNode(3, "C"),
                (0, 0): DerivationTreeNode(4, "A"),
                (0, 0, 0): DerivationTreeNode(5, "B"),
                (0, 0, 1): DerivationTreeNode(6, "D"),
            },
        )

    def test_traverse_preorder(self):
        dtree = DerivationTree(
            init_map={
                (): DerivationTreeNode(1, "A"),
                (0,): DerivationTreeNode(2, "B"),
                (1,): DerivationTreeNode(3, "C"),
                (0, 0): DerivationTreeNode(4, "A"),
                (0, 0, 0): DerivationTreeNode(5, "B"),
                (0, 0, 1): DerivationTreeNode(6, "D"),
            },
        )

        def action(path: Path, _):
            paths.append(path)

        paths = []
        dtree.traverse(
            action, kind=DerivationTree.TRAVERSE_PREORDER, reversed_child_order=False
        )
        self.assertEqual([(), (0,), (0, 0), (0, 0, 0), (0, 0, 1), (1,)], paths)

        paths = []
        dtree.traverse(
            action, kind=DerivationTree.TRAVERSE_PREORDER, reversed_child_order=True
        )
        self.assertEqual(
            list([(), (1,), (0,), (0, 0), (0, 0, 1), (0, 0, 0)]),
            paths,
        )

        paths = []
        dtree.traverse(
            action,
            abort_condition=lambda path, _: path == (0, 0, 0),
            kind=DerivationTree.TRAVERSE_PREORDER,
            reversed_child_order=False,
        )
        self.assertEqual([(), (0,), (0, 0)], paths)

    def test_traverse_postorder(self):
        dtree = DerivationTree(
            init_map={
                (): DerivationTreeNode(1, "F"),
                (0,): DerivationTreeNode(2, "B"),
                (1,): DerivationTreeNode(3, "G"),
                (0, 0): DerivationTreeNode(4, "A"),
                (0, 1): DerivationTreeNode(5, "D"),
                (0, 1, 0): DerivationTreeNode(6, "C"),
                (0, 1, 1): DerivationTreeNode(7, "E"),
                (1, 0): DerivationTreeNode(8, "I"),
                (1, 0, 0): DerivationTreeNode(9, "H"),
            },
        )

        def action(path: Path, _):
            paths.append(path)

        paths = []
        dtree.traverse(
            action, kind=DerivationTree.TRAVERSE_POSTORDER, reversed_child_order=False
        )
        self.assertEqual(
            [
                (0, 0),
                (0, 1, 0),
                (0, 1, 1),
                (0, 1),
                (0,),
                (1, 0, 0),
                (1, 0),
                (1,),
                (),
            ],
            paths,
        )

        paths = []
        dtree.traverse(
            action, kind=DerivationTree.TRAVERSE_POSTORDER, reversed_child_order=True
        )
        self.assertEqual(
            [
                (1, 0, 0),
                (1, 0),
                (1,),
                (0, 1, 1),
                (0, 1, 0),
                (0, 1),
                (0, 0),
                (0,),
                (),
            ],
            paths,
        )

        paths = []
        dtree.traverse(
            action,
            abort_condition=lambda path, _: path == (0, 1),
            kind=DerivationTree.TRAVERSE_POSTORDER,
            reversed_child_order=False,
        )
        self.assertEqual(
            [
                (0, 0),
                (0, 1, 0),
                (0, 1, 1),
            ],
            paths,
        )

    def test_traverse_bfs(self):
        dtree = DerivationTree(
            init_map={
                (): DerivationTreeNode(1, "A"),
                (0,): DerivationTreeNode(2, "B"),
                (1,): DerivationTreeNode(3, "C"),
                (0, 0): DerivationTreeNode(4, "A"),
                (0, 0, 0): DerivationTreeNode(5, "B"),
                (0, 0, 1): DerivationTreeNode(6, "D"),
                (1, 0): DerivationTreeNode(7, "F"),
                (1, 0, 0): DerivationTreeNode(8, "G"),
            },
        )

        def action(path: Path, _):
            paths.append(path)

        paths = []
        dtree.bfs(action)
        self.assertEqual(
            [(), (0,), (1,), (0, 0), (1, 0), (0, 0, 0), (0, 0, 1), (1, 0, 0)], paths
        )

        paths = []
        dtree.bfs(action, abort_condition=lambda path, _: path == (0, 0, 0))
        self.assertEqual([(), (0,), (1,), (0, 0), (1, 0)], paths)

    def test_traverse(self):
        tree = DerivationTree.from_parse_tree(
            ("1", [("2", [("4", [])]), ("3", [("5", [("7", [])]), ("6", [])])])
        )

        visited_nodes: List[int] = []

        def action(path, node):
            visited_nodes.append(int(node.value))

        visited_nodes.clear()
        tree.bfs(action)
        self.assertEqual([1, 2, 3, 4, 5, 6, 7], visited_nodes)

        visited_nodes.clear()
        tree.traverse(
            action, kind=DerivationTree.TRAVERSE_POSTORDER, reversed_child_order=False
        )
        self.assertEqual([4, 2, 7, 5, 6, 3, 1], visited_nodes)

        visited_nodes.clear()
        tree.traverse(
            action, kind=DerivationTree.TRAVERSE_POSTORDER, reversed_child_order=True
        )
        self.assertEqual([6, 7, 5, 3, 4, 2, 1], visited_nodes)

        visited_nodes.clear()
        tree.traverse(
            action, kind=DerivationTree.TRAVERSE_PREORDER, reversed_child_order=False
        )
        self.assertEqual([1, 2, 4, 3, 5, 7, 6], visited_nodes)

        visited_nodes.clear()
        tree.traverse(
            action, kind=DerivationTree.TRAVERSE_PREORDER, reversed_child_order=True
        )
        self.assertEqual([1, 3, 6, 5, 7, 2, 4], visited_nodes)

        def check_path(path, node):
            self.assertEqual(node, tree.get_subtree(path).node())

        tree.traverse(
            check_path, kind=DerivationTree.TRAVERSE_PREORDER, reversed_child_order=True
        )
        tree.traverse(
            check_path,
            kind=DerivationTree.TRAVERSE_PREORDER,
            reversed_child_order=False,
        )
        tree.traverse(
            action, kind=DerivationTree.TRAVERSE_POSTORDER, reversed_child_order=True
        )
        tree.traverse(
            action, kind=DerivationTree.TRAVERSE_POSTORDER, reversed_child_order=False
        )

    def test_postorder_traversal_for_parse_tree_conversion(self):
        parse_tree = ("1", [("2", [("4", [])]), ("3", [("5", [("7", [])]), ("6", [])])])
        tree = DerivationTree.from_parse_tree(parse_tree)
        self.assertEqual(parse_tree, traversal_to_parse_tree(tree))

    def test_terminals_and_nonterminals(self):
        dtree = DerivationTree.from_parse_tree(
            (
                "<start>",
                [
                    (
                        "<stmt>",
                        [
                            (
                                "<assgn>",
                                [
                                    ("<var>", [("x", [])]),
                                    (" := ", []),
                                    ("<rhs>", [("<digit>", [("1", [])])]),
                                ],
                            ),
                            (" ; ", []),
                            ("<stmt>", [("<assgn>", None)]),
                        ],
                    )
                ],
            )
        )

        self.assertEqual({"x", " := ", "1", " ; "}, dtree.terminals())
        self.assertEqual(
            {"<start>", "<stmt>", "<assgn>", "<var>", "<rhs>", "<digit>"},
            dtree.nonterminals(),
        )

    def test_next_path(self):
        dtree = DerivationTree(
            init_map={
                (): DerivationTreeNode(1, "A"),
                (0,): DerivationTreeNode(2, "B"),
                (0, 0): DerivationTreeNode(4, "A"),
                (0, 0, 0): DerivationTreeNode(5, "B"),
                (0, 0, 1): DerivationTreeNode(6, "D"),
                (1,): DerivationTreeNode(3, "C"),
                (1, 0): DerivationTreeNode(7, "F"),
                (1, 0, 0): DerivationTreeNode(8, "G"),
            },
        )

        next_path = ()
        paths = [()]
        while True:
            next_path = dtree.next_path(next_path)
            if next_path is None:
                break
            else:
                paths.append(next_path)

        self.assertEqual(list(dtree.paths().keys()), paths)

        self.assertEqual((1,), dtree.next_path((0, 0), skip_children=True))
        self.assertEqual((0, 0, 1), dtree.next_path((0, 0, 0), skip_children=True))
        self.assertEqual(None, dtree.next_path((), skip_children=True))

    def test_new_ids(self):
        dtree = DerivationTree(
            init_map={
                (): DerivationTreeNode(1, "A"),
                (0,): DerivationTreeNode(2, "B"),
                (0, 0): DerivationTreeNode(4, "A"),
                (0, 0, 0): DerivationTreeNode(5, "B"),
                (0, 0, 1): DerivationTreeNode(6, "D"),
                (1,): DerivationTreeNode(3, "C"),
                (1, 0): DerivationTreeNode(7, "F"),
                (1, 0, 0): DerivationTreeNode(8, "G"),
            },
        )

        new_tree = dtree.new_ids()
        ids = {node.node_id for node in new_tree.paths().values()}
        self.assertTrue(all(node_id > 8 for node_id in ids))
        self.assertTrue(new_tree.structurally_equal(dtree))
        self.assertNotEqual(new_tree, dtree)

    def test_substitute(self):
        tree = DerivationTree.from_parse_tree(
            ("1", [("2", [("4", [])]), ("3", [("5", [("7", [])]), ("6", [])])])
        )

        result = tree.substitute(
            {
                tree.get_subtree((0, 0)): DerivationTree.from_parse_tree(
                    ("8", [("9", [])])
                ),
                tree.get_subtree((1, 1)): DerivationTree.from_parse_tree(("10", [])),
            }
        )

        self.assertEqual(
            (
                "1",
                [("2", [("8", [("9", [])])]), ("3", [("5", [("7", [])]), ("10", [])])],
            ),
            result.to_parse_tree(),
        )

    def test_potential_prefix(self):
        potential_prefix_tree = DerivationTree.from_parse_tree(
            (
                "<xml-tree>",
                [
                    ("<xml-open-tag>", [("<", []), ("<id>", None), (">", [])]),
                    ("<xml-tree>", None),
                    ("<xml-close-tag>", [("</", []), ("<id>", None), (">", [])]),
                ],
            )
        )
        other_tree = DerivationTree.from_parse_tree(
            (
                "<xml-tree>",
                [
                    ("<xml-open-tag>", None),
                    ("<xml-tree>", None),
                    ("<xml-close-tag>", None),
                ],
            )
        )

        self.assertTrue(other_tree.is_prefix(potential_prefix_tree))
        self.assertFalse(potential_prefix_tree.is_prefix(other_tree))

        self.assertTrue(potential_prefix_tree.is_potential_prefix(other_tree))
        self.assertTrue(other_tree.is_potential_prefix(potential_prefix_tree))

    def test_to_dot(self):
        dtree = DerivationTree(
            init_map={
                "\x01": DerivationTreeNode(node_id=1, value="<xml-tree>"),
                "\x01\x02": DerivationTreeNode(node_id=7, value="<xml-open-tag>"),
                "\x01\x02\x02": DerivationTreeNode(node_id=10, value="<"),
                "\x01\x02\x03": DerivationTreeNode(node_id=9, value="<id>"),
                "\x01\x02\x04": DerivationTreeNode(node_id=8, value=">"),
                "\x01\x03": DerivationTreeNode(node_id=6, value="<xml-tree>"),
                "\x01\x04": DerivationTreeNode(node_id=2, value="<xml-close-tag>"),
                "\x01\x04\x02": DerivationTreeNode(node_id=5, value="</"),
                "\x01\x04\x03": DerivationTreeNode(node_id=4, value="<id>"),
                "\x01\x04\x04": DerivationTreeNode(node_id=3, value=">"),
            },
            open_leaves={(1,), (0, 1), (2, 1)},
        )

        expected = r"""// Derivation Tree
digraph {
    node [shape=plain]
    1 [label=<&lt;xml-tree&gt; <FONT COLOR="gray">(1)</FONT>>]
    1 -> 7
    1 -> 6
    1 -> 2
    7 [label=<&lt;xml-open-tag&gt; <FONT COLOR="gray">(7)</FONT>>]
    7 -> 10
    7 -> 9
    7 -> 8
    10 [label=<&lt; <FONT COLOR="gray">(10)</FONT>>]
    9 [label=<&lt;id&gt; <FONT COLOR="gray">(9)</FONT>>]
    8 [label=<&gt; <FONT COLOR="gray">(8)</FONT>>]
    6 [label=<&lt;xml-tree&gt; <FONT COLOR="gray">(6)</FONT>>]
    2 [label=<&lt;xml-close-tag&gt; <FONT COLOR="gray">(2)</FONT>>]
    2 -> 5
    2 -> 4
    2 -> 3
    5 [label=<&lt;/ <FONT COLOR="gray">(5)</FONT>>]
    4 [label=<&lt;id&gt; <FONT COLOR="gray">(4)</FONT>>]
    3 [label=<&gt; <FONT COLOR="gray">(3)</FONT>>]
}
""".replace(
            "    ", "\t"
        )

        self.assertEqual(expected, str(dtree.to_dot()))


def traversal_to_parse_tree(tree: DerivationTree) -> ParseTree:
    stack: List[ParseTree] = []

    def action(path, node: DerivationTreeNode) -> None:
        if tree.children(path) is None:
            stack.append((node.value, None))
        elif not tree.children(path):
            stack.append((node.value, []))
        else:
            children: List[ParseTree] = []
            for _ in range(len(tree.children(path))):
                children.append(stack.pop())
            stack.append((node.value, children))

    tree.traverse(
        action, kind=DerivationTree.TRAVERSE_POSTORDER, reversed_child_order=True
    )

    assert len(stack) == 1
    return stack.pop()


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
