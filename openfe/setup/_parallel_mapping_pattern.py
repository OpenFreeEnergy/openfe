from tqdm.auto import tqdm
import functools
import multiprocessing as mult

from gufe import SmallMoleculeComponent
from gufe import AtomMapper, AtomMapping

def thread_mapping(args)->list[AtomMapping]:
    '''
    Helper function working as thread for parallel execution.

    Parameters
    ----------
    args:
        contains a list of: jobID, compound_pairs, mapper, scorer

    Returns
    -------
    list[AtomMapping]:
        return a list of scored atom mappings

    '''
    jobID, compound_pairs, mapper, scorer = args
    mapping_generator = [next(mapper.suggest_mappings(
        compound_pair[0], compound_pair[1])) for
        compound_pair in compound_pairs]

    if scorer:
        mappings = [mapping.with_annotations(
            {'score': scorer(mapping)})
            for mapping in mapping_generator]
    else:
        mappings = list(mapping_generator)

    return mappings


def _parallel_map_scoring(possible_edges: list[tuple[SmallMoleculeComponent,
SmallMoleculeComponent]],
                          scorer:callable, mapper:AtomMapper,
                          n_processes:int,
                          show_progress:bool=True)->list[AtomMapping]:
    """
    This helper function parallelize mapping and scoring of a given list of
    molecule pairs.

    Parameters
    ----------
    possible_edges: tuple[SmallMoleculeComponent, SmallMoleculeComponent]
        two  molecules to be mapped.
    scorer: callable
        scoring the mappings
    mapper: AtomMapper
        atom mapper for the mappings
    n_processes: int
        number of processes for parallelization
    show_progress: bool
        show a tqdm progressbar.

    Returns
    -------
    list[AtomMapping]:
        return a list of scored atom mappings
    """
    if show_progress is True and n_processes > 1:
        n_batches = 10 * n_processes
        progress = functools.partial(tqdm, total=n_batches, delay=1.5,
                                     desc="Mapping")
    else:
        progress = lambda x: x

    possible_edges = list(possible_edges)

    # Prepare parallel execution.
    # suboptimal implementation, but itertools.batch is python 3.12,
    batches = list(possible_edges[i::n_processes] for i in
               range(n_processes))

    jobs = [(job_id, combination, mapper, scorer) for job_id,
    combination in enumerate(batches)]

    # Execute parallelism
    mappings = []
    with mult.Pool(n_processes) as p:
        for sub_result in progress(p.imap(thread_mapping, jobs)):
            mappings.extend(sub_result)

    return mappings
