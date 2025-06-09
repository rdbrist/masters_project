import pandas as pd
from datetime import timedelta


def apply_and_filter_by_offsets(
        offsets_df: pd.DataFrame = None,
        interim_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Applies the offsets from the offsets_df to the
    :param offsets_df: Dataframe of offsets with id as index and an integer for
        the offset to apply to all timestamps for that person.
    :param interim_df: Dataframe to which the offsets have to be applied.
    :return: Dataframe with the same shape, with timestamps offset, and limited
        to only those ids that exist in both.
    """
    if offsets_df.index.duplicated().any():
        raise ValueError("Profile offsets DataFrame contains duplicate IDs."
                         " Please ensure each ID is unique such that only"
                         " one offset exists.")
    zip_ids = offsets_df.index.unique()

    # Check for missing ids before mapping
    missing_ids = (
            set(interim_df.index.get_level_values('id')) -
            set(offsets_df.index))
    if missing_ids:
        raise ValueError(f"IDs missing in offsets_df: {missing_ids}")

    interim_df = interim_df.reset_index()
    interim_df['offset'] = interim_df['id'].map(offsets_df['offset'])
    interim_df['datetime'] += interim_df['offset'].apply(timedelta)
    interim_df['day'] = interim_df['datetime'].dt.date
    interim_df['time'] = interim_df['datetime'].dt.time
    return interim_df.set_index(['id', 'datetime']).sort_index()

    

