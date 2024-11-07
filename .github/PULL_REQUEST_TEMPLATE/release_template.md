<!--
Checklist for releasing a new version of openfe. 
-->

Make the PR:
* [ ] Create a new release-prep branch corresponding to the version name, e.g. `release-prep-v1.2.0`.  Note: please follow [semantic versioning](https://semver.org/).
* [ ] Check that all user-relevant updates are included in the `news/` rever `.rst` files. You can backfill any additional items by making a new .rst, e.g. `backfill.rst`
* [ ] Run [rever](https://regro.github.io/rever-docs/index.html#), e.g. `rever 1.2.0`. This will auto-commit `docs/CHANGELOG.md` and remove the `.rst` files from `news/`. 
* [ ] Verify that`docs/CHANGELOG.md` looks correct.
* [ ] Make the PR and verify that CI/CD passes. 
* [ ] Merge the PR into `main`.

After Merging the PR:
* [ ] Make a new [release draft through github](https://github.com/OpenFreeEnergy/openfe/releases/new)
* [ ] Add the link to the [openfe website changelog](https://docs.openfree.energy/en/stable/CHANGELOG.html) in the github release notes.
* [ ] Wait for PR to be auto-created on [conda-forge openfe-feedstock](https://github.com/conda-forge/openfe-feedstock), verify tests pass, and merge.

## Developers certificate of origin
- [ ] I certify that this contribution is covered by the MIT License [here](https://github.com/OpenFreeEnergy/openfe/blob/main/LICENSE) and the **Developer Certificate of Origin** at <https://developercertificate.org/>.
