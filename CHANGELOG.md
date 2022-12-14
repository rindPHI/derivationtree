# Changelog

This file contains the notable changes since version 0.1.0.

## [unreleased]

## [0.4.3]

### Changed

- `DerivationTree.paths`,`DerivationTree.leaves`, and `DerivationTree.open_leaves` now
  return generators instead of whole dictionaries for better performance.

## [0.4.2]

### Changed

- Fixed `DerivationTree.is_valid_path`
- Path objects are memoized
- Potential performance improvement in `DerivationTree.replace_subtree`
- Performance improvement in `DerivationTree.__init__`

## [0.4.1]

### Changed

- Implemented `PathIterator.__getitem__` to enable, e.g., unpacking of Paths

## [0.4.0]

### Changed

- Using new `Path` objects instead of tuples for paths. This avoids the necessity to
  convert between datrie string keys and numeric paths, the cost of which accumulates
  over time.
- More efficient `DerivationTree.replace_trie`
- More efficient `DerivationTree.leaves`
- More efficient `DerivationTree.find_node`
- Fixed `DerivationTree.structural_hash`
- Fixed `DerivationTree.is_potential_prefix`

## [0.3.3]

### Changed

- Fix in `DerivationTree.to_string` method; now, not displaying (closed) nonterminal
  leaves
- Don't use deep copy in `DerivationTree.replace_path`, but create new datrie object.
  This is obviously more efficient. However, there should be a more (time & memory)
  efficient way of replacing a leaf in a tree than copying the whole datrie...
- More efficient `DerivationTree.__eq__` and `DerivationTree.is_leaf` methods
- `DerivationTree.paths` can return mappings from datrie keys to nodes if requested 

## [0.3.2]

### Changed

- Made method `DerivationTree.trie_from_init_map` private (`__` prefix)
- Method `DerivationTree.is_complete` now returns True iff the whole *tree*, and not
  only the root (or otherwise specified) node is not open.
- Method `DerivationTree.filter` now returns a dict instead of a list of pairs.
- Bug fix in `DerivationTree.replace_path` for trees with a non-trivial root path.

## [0.3.1]

### Changed

- `DerivationTree.from_node` infers optional `is_open` parameter from whether the given
  value is a nonterminal or not.
 
## [0.3.0]

### Changed

- Changed signature of `DerivationTree.children` method, which now returns a dictionary
  of paths and nodes instead of only nodes.

## [0.2.0]

### Added

- Added convenience methods `DerivationTree.from_node` and `DerivationTree.add_children`
  for constructing a derivation tree from a value, optional id, and "open" flag, and
  for adding children to an existing derivation tree leaf.

### Changed

- Preventing replacement of paths that do not have an immediate parent in the tree.
  One *can* replace a path that is a child of a previous leaf node in the tree, but
  did not exist in the tree before.
