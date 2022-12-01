DerivationTree: An Efficient, Trie-Based Derivation Tree Implementation
=======================================================================

[![Python](https://img.shields.io/pypi/pyversions/derivationtree.svg)](https://pypi.python.org/pypi/derivationtree/)
[![Version](http://img.shields.io/pypi/v/derivationtree.svg)](https://pypi.python.org/pypi/derivationtree/)
[![Build Status](https://img.shields.io/github/workflow/status/rindPHI/derivationtree/Test%20DerivationTree)](https://github.com/rindPHI/derivationtree/actions/workflows/test-derivationtree.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**TODO**

## Build, Run, Install

DerivationTree requires **Python 3.10**.

### Install

Usually, a simple `pip install derivationtree` should suffice.
We recommend installing DerivationTree inside a virtual environment (virtualenv):

```shell
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install derivationtree
```

### Build 

DerivationTree is built locally as follows:

```shell
git clone https://github.com/rindPHI/derivationtree.git
cd derivationtree/

python3.10 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install --upgrade build
python3 -m build
```

Then, you will find the built wheel (`*.whl`) in the `dist/` directory.

### Testing & Development

For development, we recommend using DerivationTree inside a virtual environment (
virtualenv). By thing the following steps in a standard shell (bash), one can run the
ISLa tests:

```shell
git clone https://github.com/rindPHI/derivationtree.git
cd derivationtree/

python3.10 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements_test.txt

# Run tests
pip install -e .[dev,test]
python3 -m pytest tests
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## Copyright, Authors and License

Copyright © 2022 [CISPA Helmholtz Center for Information Security](https://cispa.de/en).

The DerivationTree code and documentation was, unless otherwise indicated, authored by
[Dominic Steinhöfel](https://www.dominic-steinhoefel.de).

DerivationTree is released under the GNU General Public License v3.0 (
see [COPYING](COPYING)).