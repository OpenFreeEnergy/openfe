<!--
Checklist for releasing a new version of openfe. 
-->

Make the PR:
* [ ] Create a new release prep branch corresponding to the version name, e.g. `release/v1.2.0`.  Note: please follow [semantic versioning](https://semver.org/).
* [ ] Check that all user-relevant updates are included in the `news/` rever `.rst` files. You can backfill any additional items by making a new .rst, e.g. `backfill.rst`
* [ ] Run [rever](https://regro.github.io/rever-docs/index.html#), e.g. `rever 1.2.0`. This will auto-commit `docs/CHANGELOG.md` and remove the `.rst` files from `news/`. 
* [ ] Verify that`docs/CHANGELOG.rst` looks correct and that it renders as expected in the docs preview.
* [ ] If needed, create a release of the [example notebooks repository](https://github.com/OpenFreeEnergy/ExampleNotebooks) and update the pinned release version in the `openfe/docs/conf.py`.
* [ ] Make the PR and verify that CI/CD passes. 
  * [ ] [feedstock packaging tests](https://github.com/OpenFreeEnergy/openfe/actions/workflows/release-prep-feedstock.yaml)
  * [ ] [example notebooks](https://github.com/OpenFreeEnergy/openfe/actions/workflows/release-prep-examplenotebooks.yaml)
  * [ ] [GPU tests](https://github.com/OpenFreeEnergy/openfe/actions/workflows/aws-gpu-integration-tests.yaml)
* [ ] Merge the PR into `main`.

After Merging the PR [follow this guide](https://github.com/OpenFreeEnergy/openfe/wiki/How-to-create-a-new-release)

