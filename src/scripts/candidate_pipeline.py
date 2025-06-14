from loguru import logger

from src.configurations import Configuration, FifteenMinute
from src.helper import separate_flat_file, filter_separated_by_ids
from src.candidate_selection import (remove_null_variable_individuals,
                                     provide_data_statistics,
                                     plot_nights_vs_avg_intervals,
                                     Nights,
                                     reconsolidate_flat_file_from_nights)
from src.data_processing.read_preprocessed_df import (apply_and_filter_by_offsets,
                                                      ReadPreprocessedDataFrame)
from src.resample import resample_to_30_minute_intervals
from src.data_processing.read import read_profile_offsets_csv


def main():
    # 0. Import the preprocessed data
    fifteen_minute = FifteenMinute()
    config = Configuration()
    new_sample_rate = 30  # Resampling to 30-minute intervals
    df = ReadPreprocessedDataFrame(fifteen_minute, file_type='parquet').df
    df_offsets = read_profile_offsets_csv(config)

    # 1. Resample to 30-minute intervals
    df = resample_to_30_minute_intervals(df)

    # 2. Apply offsets to the data
    df_processed = apply_and_filter_by_offsets(offsets_df=df_offsets,
                                               interim_df=df)

    # 3. Remove individuals with null variables
    df_processed = remove_null_variable_individuals(df_processed)

    # 4. Separate the data into individual dataframes
    separated = separate_flat_file(df_processed)

    # 5. Process the data through the Nights class
    df_overall_stats = provide_data_statistics(separated,
                                               sample_rate=new_sample_rate)

    # 6. Aggregate stats and visualise the data
    plot_nights_vs_avg_intervals(df_overall_stats)

    # 7. Identify individuals with satisfactory level of completeness
    df_filtered = df_overall_stats[df_overall_stats['complete_nights'] > 30]

    logger.info(
        f'Number of individuals with > 30 complete nights: {len(df_filtered)}')
    candidates = df_filtered.index.tolist()
    logger.info(candidates)

    # 8. Get only the complete nights for the candidates for further analysis
    filtered_separated = filter_separated_by_ids(separated, candidates)
    nights_objects = []
    for id_, df in filtered_separated:
        nights = Nights(zip_id=id_, df=df, sample_rate=new_sample_rate)
        nights_objects.append(nights.remove_incomplete_nights())
        logger.info(f'Candidate: {id_}, Complete Nights: '
                    f'{nights.overall_stats["complete_nights"]}')

    df_all_selected = reconsolidate_flat_file_from_nights(nights_objects)


if __name__ == "__main__":
    main()
