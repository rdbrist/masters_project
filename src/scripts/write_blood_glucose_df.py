from src.configurations import Configuration
from src.data_processing.read import read_all_bg
from src.data_processing.write import write_read_record
import time
from datetime import timedelta

# set as_flat_file to True if you want to save one big flat bg csv or set it to false if you want a bg_csv per id
def main():
    start_time = time.time()
    config = Configuration()
    as_flat_file = config.as_flat_file
    folder = config.data_folder if as_flat_file else config.perid_data_folder
    result = read_all_bg(config)
    write_read_record(result, as_flat_file, folder, config.bg_file, file_type='csv')
    write_read_record(result, as_flat_file, folder, config.bg_file, file_type='parquet')
    print(f'Execution time: {timedelta(seconds=(time.time() - start_time))}')


if __name__ == "__main__":
    main()
