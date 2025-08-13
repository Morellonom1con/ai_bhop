"""Helper functions for schema handling """


def clean_schema_addresses(list_, udf_stages=None):
    """Clean a address from the addtitionnal layers of SCHEMA.

    Used only when a SCHEMA address must be found in the data to validate

     Parameters :
    -----------
    list_ : a list of string
        address in a nested dict
    udf_stages : a list of additionnal user defined stages (udf)
    Returns :
    ---------
    list
        the same list without SCHEMA intermedaite stages
    """
    skipped_stages = ["properties", "oneOf"]
    if udf_stages is not None:
        for udf_stage in udf_stages:
            skipped_stages.append(udf_stage)
    out = []
    for item in list_:
        if item not in skipped_stages:
            out.append(item)
    return out
