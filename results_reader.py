import json
import properties as prop



def read_results(prefix='', results_file=prop.RESULTS_FILE):
    with open(prefix + results_file, 'a+') as f:
        f.seek(0)
        try:
            results = json.load(f)
        except ValueError:
            results = {}

    return results


def set_results(results, results_file=prop.RESULTS_FILE):
    with open(results_file, 'w') as f:
        json.dump(results, f)

