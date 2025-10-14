# Contributing to OpenFE

We welcome any fixes or code contributions to the **openfe** and the larger OpenFE ecosystem.
Note that any contributions made must be made under a MIT license.

Please read our [code of conduct](Code_of_Conduct.md) to understand the standards you must adhere to.

## Installing a Dev Environment

It's common to need to develop openfe and gufe in tandem, to do so:
- gufe is strictly upstream of openfe, meaning that all gufe code should be test in gufe.

- mamba create -f env.yaml in openfe
- pip install -e . # in openfe/dev_branch
- pip install -e . # in gufe/dev_branch

to make CI test the environments properly;
- in openfe/env.yaml and openfe/docs/env.yaml, pin gufe to your dev working branch

once you're ready to merge:
- merge in gufe PR
- switch openfe env.yamls back to gufe@main, then push


## Contributing a new feature (multi-PR contributing)
- open an issue describing the feature you want to contribute
    - use sub issues for larger, more complex project. You are encouraged to add sub issues throughout the development process. Try to aim for 1 PR per issue.
- you may want to request that maintainers create a designated branch for large features that require multiple PRs
    - this way, you can work off of a fork, with this feat/ branch as the target.
- make a feat/my-new-feature branch that will be long-lived branch
- break your work into smaller issue/PR pairs that target `feat/my-new-feature`, and ask for frequent reviews.
- PR title should be essentially a changelog entry

## CI, linting, style guide
- CI tests, docs, etc, will run on every PR that as `main` as a target.
- pre-commit comment
- rebase or merge, we don't care, but keep your branch up to date with `main`
- SQUASH ALWAYS 🍠
- [numpy docstrings](https://numpydoc.readthedocs.io/en/latest/format.html), use type hinting as much as is reasonable.
- tests in every PR
- using `precommit` locally:
    - pyproject.toml
- any pins to versions in environments etc. need to have a comment explaining why, linking to the related issue, etc.

## Review process
- In general, push PRs early so that work is visibile, but leave a PR in draft-mode until you want it to be reviewed by the team. Once it's marked as "ready for review," the OpenFE team will assign a reviewer.

# TODO: ai use disclosure