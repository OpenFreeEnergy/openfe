# Contributing to OpenFE

We welcome fixes and code contributions to **openfe** and the [larger OpenFE ecosystem](https://github.com/OpenFreeEnergy).
Since we practice issue-driven development, please **open an issue to discuss a proposed contribution before making a Pull Request (PR)**.

Note that any contributions made must be made under a [MIT license](https://opensource.org/license/mit).

Please read our [code of conduct](Code_of_Conduct.md) to understand the standards you must adhere to.

## Setting up a Development Environment

See the instructions for [installing a local dev environment](https://docs.openfree.energy/en/latest/installation.html#developer-install) to get started.


It's common to need to develop **openfe** in tandem with another OpenFE Ecosystem package.
For example, you may want to add some feature to [**gufe**](https://github.com/OpenFreeEnergy/gufe), and make sure that its functionality works as intended with **openfe**.

To build dev versions of both packages locally, follow the above instructions for installing **openfe** for development, then (using **gufe** as an example):

``` bash
mamba activate openfe_env  # if not already activated
git clone git@github.com:OpenFreeEnergy/gufe.git
cd gufe/
pip install -e . --no-deps
```

To make sure the Continuous Integration (CI) tests run properly on your PR, in `openfe/env.yaml` and `openfe/docs/env.yaml`, pin **gufe** to your dev working branch

```yaml
...

  - pip:
    - git+https://github.com/OpenFreeEnergy/gufe@feat/your-dev-branch
```
Once tests pass, the PR is approved, and you're ready to merge:
  1. merge in the **gufe** PR
  2. switch `openfe/env.yaml` and `openfe/docs/env.yaml` back to `gufe@main`, then re-run CI.

## Contribution Guidelines

### Issue-driven development

OpenFE adheres to issue-driven development, especially for external contributions. 
This saves time for both the contributor and OpenFE maintainers, as project scope and planned work are decided _before_ implementation begins.

In other words, **before starting a pull request, [open an issue](https://github.com/OpenFreeEnergy/openfe/issues)**.

A good issue should:
- define the problem, whether that is a lack of behavior(feature addition), or broken behavior (bugfix).
- optionally outline a proposed solution - multiple possible solutions are encouraged.

Templates are provided when opening an issue to act as a guideline for best practices.


### Test-driven development

Scientific validity is at the core of OpenFE's development process, and it is critical that the code author and reviewer(s) have a shared understanding of what science the code contribution is meant to accomplish.
To this end, we strongly encourage test-driven development, with tests defined and written by a developer who understands (and can clearly explain) the scientific purpose of the code.

<!-- Ensuring scientific validity -->

### Use semantic branch names
We encourage starting branch names with one of the following "types", followed by a short description or issue number.
  - `feat/`: new user-facing feature
  - `fix/`: bugfixes
  - `docs/`: changes to documentation only
  - `refactor/`: refactoring that does not change behavior
  - `ci/`: changes to CI workflows

<!-- - PR titles should be essentially a changelog entry. TODO: make this clearer, maybe examples -->

### CI, linting, style guide

All PRs come templated with a checklist of the highest priority criteria, along with CI that runs tests, linting, and other relevant checks that should all pass before a PR is merged.

- Use [numpy docstrings](https://numpydoc.readthedocs.io/en/latest/format.html)
- Add a news item for any user-facing changes.
- Rebase or merge, we don't care, but keep your branch up to date with `main`.
- Always "squash and merge" when merging pull requests into main to keep the git log readable.
  This also makes it easier to use the `git bisect` tool.
  Squash and merge is enforced by default on most OpenFE projects.
- Use [type hints](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html), especially for new code.
- Every PR should have tests that cover changes. You will see the coverage report in CI.
- Any pins to versions in `environment.yaml` etc. need to have a comment explaining the purpose of the pin and/or a link to the related issue.

<!-- TODO: add info about pre-commit comment -->
<!-- TODO: using `precommit` locally: pyproject.toml -->


### Review process
- In general, push PRs early so that work is visible, but leave a PR in draft-mode until you want it to be reviewed by the team.
  Once it's marked as "ready for review," the OpenFE team will assign a reviewer.

### Larger contributions (multi-PR contributions)
- [Open an issue](https://github.com/OpenFreeEnergy/openfe/issues) describing the feature you want to contribute and get feedback from the OpenFE team.
  Use sub-issues to break down larger, more complex projects.
    - You are encouraged to add sub issues throughout the development process.
    - Try to aim for 1 PR per issue
- You may want to request that maintainers create a designated branch for large features that require multiple PRs.
  This way, the `feat/` branch can be the target of several smaller PRs.
- Break your work into smaller issue/PR pairs that target `feat/my-new-feature`, and ask for frequent reviews.


## Generative AI Use Policy

We care about promoting new contributions and are willing to spend time reviewing code from new or inexperienced developers.
However, with generative AI, a large volume of code can be generated quickly, pushing the burden onto reviewers.


Furthermore, test driven development (with tests defined and written _by the code author_), thoughtful conversation on Issues and PRs, and thorough code review are how we prioritize scientific validity and code quality.

The OpenFE core developers reserve the right to reject contributions if we judge that they do not follow the spirit of these principles.```

<!-- this is subject to change, as the team is continually reevaluating  -->

OpenFE's guidelines around generative AI are based on the principles described above, and can be summarized as:

1. **Own your code**: No PRs authored by agentic bots will be accepted. A human author must take responsibility for all issues and PRs.
2. **Prioritize your reviewer**: OpenFE maintainers review all PRs, meaning that a human reviews every line that is committed.
   PRs should be constrained to a size and scope that is reasonable for a human to review thoroughly.