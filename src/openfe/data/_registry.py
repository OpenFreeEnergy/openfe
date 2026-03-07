import pooch

POOCH_CACHE = pooch.os_cache("openfe")

zenodo_rfe_simulation_nc = dict(
    base_url="doi:10.5281/zenodo.15375081/",
    fname="simulation.nc",
    known_hash="md5:bc4e842b47de17704d804ae345b91599",
)
zenodo_t4_lysozyme_traj = dict(
    base_url="doi:10.5281/zenodo.15212342",
    fname="t4_lysozyme_trajectory.zip",
    known_hash="sha256:e985d055db25b5468491e169948f641833a5fbb67a23dbb0a00b57fb7c0e59c8",
)
zenodo_industry_benchmark_systems = dict(
    base_url="doi:10.5281/zenodo.15212342",
    fname="industry_benchmark_systems.zip",
    known_hash="sha256:2bb5eee36e29b718b96bf6e9350e0b9957a592f6c289f77330cbb6f4311a07bd",
)
zenodo_resume_data = dict(
    base_url="doi:10.5281/zenodo.18331259",
    fname="multistate_checkpoints.zip",
    known_hash="md5:6addeabbfa37fd5f9114e3b043bfa568",
)

zenodo_data_registry = [
    zenodo_rfe_simulation_nc,
    zenodo_t4_lysozyme_traj,
    zenodo_industry_benchmark_systems,
    zenodo_resume_data,
]
