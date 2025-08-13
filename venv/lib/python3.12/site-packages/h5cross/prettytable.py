
from prettytable import PrettyTable

from h5cross.utils import unwrap_dict


def get_stats_table(stats_dict, full_path=True, digits=6):
    """Function to generate a pretty table output of the added statistics.
    """
    unwrapped_dict = unwrap_dict(stats_dict)  # to avoid nested

    # assumes all have computed same stats
    headers = list(unwrapped_dict[list(unwrapped_dict.keys())[0]]._fields)

    table = PrettyTable(['Dataset'] + headers)

    for key, values in unwrapped_dict.items():
        if not full_path:
            key = key.split('/')[-1]

        table.add_row(_build_row(key, values, headers, digits))

    table.align['Dataset'] = "l"

    return table


def _build_row(key, values, headers, digits=6):
    return [key] + [round(getattr(values, stat), digits) for stat in headers]
