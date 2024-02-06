from tqdm.auto import tqdm
import functools
import itertools
import multiprocessing as mult

def thread_mapping(args):
    '''
    Helper function working as thread for parallel execution.
    Parameters
    ----------
    compound_pair

    Returns
    -------

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


def _parallel_map_scoring(possible_edges, scorer, mapper, n_processes,
                          show_progress=True):
    if show_progress is True and n_processes > 1:
        n_batches = 10 * n_processes
        progress = functools.partial(tqdm, total=n_batches, delay=1.5,
                                     desc="Mapping")
    else:
        progress = lambda x: x

    possible_edges = list(possible_edges)
    n_batches = 10 * n_processes
    total = len(possible_edges)

    # size of each batch +fetch division rest
    batch_num = (total // n_batches) + 1

    # Prepare parallel execution.
    jobs = [(job_id, combination, mapper, scorer) for job_id,
    combination in enumerate(itertools.batched(possible_edges, batch_num))]

    # Execute parallelism
    mappings = []
    with mult.Pool(n_processes) as p:
        for sub_result in progress(p.imap(thread_mapping, jobs)):
            mappings.extend(sub_result)

    return mappings