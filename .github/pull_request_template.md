<!--
Thank you for pull request.
Below are a few things we ask you kindly to self-check before getting a review. Remove checks that are not relevant.
-->

<!--
Please note any issues this fixes using [closing keywords]( https://help.github.com/articles/closing-issues-using-keywords/ ):
-->

<!--
see https://regro.github.io/rever-docs/news.html for details on how to add news entry (you do not need to run the rever command)
-->

Checklist
* [ ] All new code is appropriately documented (user-facing code _must_ have complete docstrings).
* [ ] Added a ``news`` entry, or the changes are not user-facing.
* [ ] Ran pre-commit: you can run [pre-commit](https://pre-commit.com) locally or comment on this PR with `pre-commit.ci autofix` before requesting review.

Manual Tests: these are slow so don't need to be run every commit, only before merging and when relevant changes are made (generally at reviewer-discretion). 
* [ ] [GPU integration tests](https://github.com/OpenFreeEnergy/openfe/actions/workflows/gpu-integration-tests.yaml)
* [ ] [example notebook testing](https://github.com/OpenFreeEnergy/openfe/actions/workflows/test-example-notebooks.yaml)
* [ ] [packaging tests](https://github.com/OpenFreeEnergy/openfe/actions/workflows/test-feedstock-pkg-build.yaml): run this for any large feature PRs or PRs that add test data.


## Developers certificate of origin
- [ ] I certify that this contribution is covered by the MIT License [here](https://github.com/OpenFreeEnergy/openfe/blob/main/LICENSE) and the **Developer Certificate of Origin** at <https://developercertificate.org/>.
