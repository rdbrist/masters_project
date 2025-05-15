from src.configurations import Configuration, Irregular
from src.config import INTERIM_DATA_DIR
from src.data_processing.preprocess import dedub_device_status_dataframes
from src.data_processing.read import read_all_device_status
from src.data_processing.write import write_read_record


# set as_flat_file to True if you want to save one big flat bg csv or set it to false if you want a bg_csv per id
# change keep_columns if you want to keep different columns in the resulting file, default IOB, COB, BG plus time and id
def main():
    config = Configuration()
    as_flat_file = config.as_flat_file
    folder = INTERIM_DATA_DIR if as_flat_file else (INTERIM_DATA_DIR / 'perid')
    result = read_all_device_status(config)
    de_dub_result = dedub_device_status_dataframes(result)

    # write irregular
    write_read_record(de_dub_result, as_flat_file, folder, Irregular.csv_file_name(),
                      keep_cols=config.keep_columns)


if __name__ == "__main__":
    main()
