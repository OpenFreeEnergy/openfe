# Contributing to OpenFE

We welcome fixes and code contributions to **openfe** and the [larger OpenFE ecosystem](https://github.com/OpenFreeEnergy).
Note that any contributions made must be made under a MIT license.

Please read our [code of conduct](Code_of_Conduct.md) to understand the standards you must adhere to.

## Installing a Development Environment

### Installing openfe for development

``` bash
mamba env create -f environment.yml
mamba activate openfe_env
python -m pip install -e --no-deps .
```

### Multi-project development
It's common to need to develop *openfe** in tandem with another OpenFE Ecosystem package.
For example, you may want to add some feature to **gufe**, and make sure that its functionality works as intended with **openfe**.

To build dev versions of both packages locally, follow the above instructions for installing **openfe** for development, then (using **gufe** as an example):

``` bash
mamba activate openfe_env  # if not already activated
git clone git@github.com:OpenFreeEnergy/gufe.git
cd gufe/
pip install -e .
```

To make sure the CI tests run properly on your PR, in `openfe/env.yaml` and `openfe/docs/env.yaml`, pin **gufe** to your dev working branch

```yaml
...

  - pip:
    - git+https://github.com/OpenFreeEnergy/gufe@feat/your-dev-branch
```
Once tests pass, the PR is approved, and you're ready to merge:
  1. merge in the **gufe** PR
  2. switch `openfe/env.yaml` and `openfe/docs/env.yaml` back to `gufe@main`, then re-run CI.

## Contribution Guidelines

### Use semantic branch names
- Start branch names with one of the following "types", followed by a short description or issue number.
    - `feat/`: new user-facing feature
    - `fix/`: bugfixes
    - `docs/`: changes to documentation only
    - `refactor/`: refactoring that does not change behavior
    - `ci/`: changes to CI workflows

<!-- - PR titles should be essentially a changelog entry. TODO: make this clearer, maybe examples -->

### CI, linting, style guide
- CI tests, docs, etc, will run on every PR that has `main` or a branch that begins with `feat/` as a target.
- Add a news item for any user-facing changes.
- Rebase or merge, we don't care, but keep your branch up to date with `main`.
- Always "squash and merge" to keep the git log readable.
- Use [numpy docstrings](https://numpydoc.readthedocs.io/en/latest/format.html)
- We encourage but do not require [type hints](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html).
- Every PR should have tests that cover changes. You will see the coverage report in CI.
- Any pins to versions in `environment.yaml` etc. need to have a comment explaining the purpose of the pin and/or a link to the related issue.

<!-- TODO: add info about pre-commit comment -->
<!-- TODO: using `precommit` locally: pyproject.toml -->


### Review process
- In general, push PRs early so that work is visible, but leave a PR in draft-mode until you want it to be reviewed by the team. Once it's marked as "ready for review," the OpenFE team will assign a reviewer.

### Larger contributions (multi-PR contributions)
- [Open an issue](https://github.com/OpenFreeEnergy/openfe/issues) describing the feature you want to contribute and get feedback from the OpenFE team. Use sub-issues to break down larger, more complex projects.
    - You are encouraged to add sub issues throughout the development process.
    - Try to aim for 1 PR per issue!
- You may want to request that maintainers create a designated branch for large features that require multiple PRs. This way, the `feat/` branch can be the target of several smaller PRs.
- Break your work into smaller issue/PR pairs that target `feat/my-new-feature`, and ask for frequent reviews.

 <!-- TODO: ai use disclosure -->