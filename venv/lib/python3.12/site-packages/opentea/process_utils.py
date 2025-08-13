"""Utilities for opentea additionnal processing in tabs"""
import os
import sys
import yaml



__all__ = ["process_tab", "update_3d_callback", "rec_fusion"]


def process_tab(func_to_call):
    """Execute the function of an external process.external.

    func_to_call : see above for a typical function
    to be called by openTea GUIS

    A typical callback scriptwill look like this:

    ::
        def template_aditional_process(nob_in):
            nob_out = nob_in.copy()
            # Your actions here to change the content of nob_out
            # nob_out["foobar"] = 2 * nob_in["foobar"]
            # (...)
            return nob_out


        if __name__ == "__main__":
            process_tab(template_aditional_process)


    """
    with open(sys.argv[1], "r") as fin:
        data = yaml.load(fin, Loader=yaml.SafeLoader)
    data_out = func_to_call(data)
    with open(".dataset_to_gui.yml", "w") as fout:
        yaml.dump(data_out, fout, default_flow_style=False)

def rec_fusion(dat_, add_dat_):
    """Recutsive function for the fusion of schemas

    For Dicts and usual leafs (strings, numbers)
    - if the data is only in dat_ > Kept
    - if the data is both in dat_ and add_dat_> Replaced
    - if the data is only in add_dat_ > Added

    For a list, add_dat_ items are concatenated after dat_, no fusion.
    """
    if isinstance(dat_, dict):
        # Replace identical keys
        for key in dat_:
            if key in add_dat_:
                dat_[key] = rec_fusion(dat_[key], add_dat_[key])
        # add additionnal keys
        for key in add_dat_:
            if key not in dat_:
                dat_[key] = add_dat_[key]

    # extend lists
    elif isinstance(dat_, list):
        for item in add_dat_:
            if item not in dat_:
                dat_.append(item)
        #dat_.extend(add_dat_)
        #dat_ = [i for n, i in enumerate(dat_) if i not in dat_[n + 1:]]
    else:
        if dat_ != add_dat_:
            dat_ = add_dat_
    return dat_
