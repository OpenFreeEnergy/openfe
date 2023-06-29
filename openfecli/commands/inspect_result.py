
def print_failure_unit_error(failure_unit):
    tb_text = failure_unit.traceback
    # TODO: add a try/except around importing pygments; if it is there,
    # let's make the output pretty
    print(tb_text)

def result_summary(result_dict, output):
    import math
    summ_lines = []
    # we were success or failure
    success = "FAILURE" if math.isnan(result_dict['estimate']) else "SUCCESS"
    summ_lines.append(f"This edge was a {success}")
    units = result_dict['unit_results']
    summ_lines.append(f"This edge consists of {len(units)}")
    for unit in ...


def inspect_result(json_filename):
    ...
