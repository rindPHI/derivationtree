# Changelog

This file contains the notable changes since version 0.1.0.

## [unreleased]

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
