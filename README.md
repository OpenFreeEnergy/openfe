[![Logo](https://img.shields.io/badge/OSMF-OpenFreeEnergy-%23002f4a)](https://openfree.energy/)
[![build](https://github.com/OpenFreeEnergy/openfe/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/OpenFreeEnergy/openfe/actions/workflows/ci.yaml)
[![coverage](https://codecov.io/gh/OpenFreeEnergy/openfe/branch/main/graph/badge.svg)](https://codecov.io/gh/OpenFreeEnergy/openfe)
[![documentation](https://readthedocs.org/projects/openfe/badge/?version=stable)](https://docs.openfree.energy/en/stable/?badge=stable)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8344248.svg)](https://doi.org/10.5281/zenodo.8344248)


# `openfe` - A Python package for executing alchemical free energy calculations.

The `openfe` package is the flagship project of [Open Free Energy](https://openfree.energy),
a pre competitive consortium aiming to provide robust, permissively licensed open source tools for molecular simulation in the drug discovery field.

Using `openfe` you can easily plan and execute alchemical free energy calculations.

See our [website](https://openfree.energy/) for more information on the project,
[try for yourself](https://try.openfree.energy) from the comfort of your browser,
and we have [documentation on using the package](https://docs.openfree.energy/en/latest/index.html).

## License

This library is made available under the [MIT](https://opensource.org/licenses/MIT) open source license.

## Install

### Latest release

The latest release of `openfe` can be installed via `mamba`, `docker`, or a `single file installer`. See [our installation instructions](https://docs.openfree.energy/en/stable/installation.html) for more details.
Dependencies can be installed via conda through:

### Development version

The development version of `openfe` can be installed directly from the `main` branch of this repository.

First install the package dependencies using `mamba`:

```bash
mamba env create -f environment.yml
```

The openfe library can then be installed via:

```
python -m pip install --no-deps .
```

## Authors

The OpenFE development team.

## Acknowledgements

OpenFE is an [Open Molecular Software Foundation](https://omsf.io/) hosted project.
