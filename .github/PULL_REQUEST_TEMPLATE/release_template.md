<!--
Checklist for releasing a new version of openfe. 
-->

release PR checklist:
* [ ] All user-relevant updates are included in the `news/` rever `.rst` files. (You can backfill any additional items by making a new .rst, e.g. `backfill.rst`.)
* [ ] [rever](https://regro.github.io/rever-docs/index.html#) has been run, e.g. `rever 1.2.0`. This will auto-commit `docs/CHANGELOG.md` and remove the `.rst` files from `news/`. 
* [ ] `docs/CHANGELOG.md` looks correct.

after merging this PR:
* Make a PR into the [example notebooks repository](https://github.com/OpenFreeEnergy/ExampleNotebooks) to update the version used in `showcase/openfe_showcase.ipynb` and `.binder/environment.yml`
* [continue following this guide](https://github.com/OpenFreeEnergy/openfe/wiki/How-to-create-a-new-release)

