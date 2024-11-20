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
* [ ] Make a PR into the [example notebooks repository](https://github.com/OpenFreeEnergy/ExampleNotebooks) to update the version used in `showcase/openfe_showcase.ipynb` and `.binder/environment.yml`

After Merging the PR [follow this guide](https://github.com/OpenFreeEnergy/openfe/wiki/How-to-create-a-new-release)

