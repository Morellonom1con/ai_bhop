"""Utility to create plots of CSV data in pure terminal environments"""

import pandas as pd
import plotille as plt


def plot_csv_data(filename, keys, delimiter=None):
    """Create a plot from a CSV file.

    The FILENAME is read with Pandas
    KEYS is a list of headers ["y/Y", "Cows", "Cats"]
    Plot KEYS[1:] with respect to KEYS[0] 

    Parameters:
    -----------
    :param filename: str, path to the csv file
    :param keys: list of str, names of the elements to plot
    :param delimiter: char, delimiter of the CSV file
    
    Returns :
    ---------
    graph: a string representing the graph
    """
    df_ = pd.read_csv(filename, delimiter=delimiter)

    try:
        fig = plt.Figure()
        fig.x_label=keys[0]

        x = df_[keys[0]].tolist() 
        for curve_key in keys[1:]:
            y = df_[curve_key].tolist() 

            fig.plot(x,y, label=curve_key)
    except KeyError as err:
        msg_err = f" : Possible keys are: {'|'.join(df_.keys())}"
        raise ValueError("Wrong columns names" + msg_err + str(err) )

    out = fig.show(legend=True)
    return out
