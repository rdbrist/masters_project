from src.configurations import Configuration
from src.data_processing.read import read_all_device_status
from src.data_processing.write import write_read_record
import time
from datetime import timedelta
from src.config import INTERIM_DATA_DIR


def main():
    start_time = time.time()
    config = Configuration()

    as_flat_file = config.as_flat_file
    folder = INTERIM_DATA_DIR if as_flat_file else (INTERIM_DATA_DIR / 'perid')
    result = read_all_device_status(config)
    write_read_record(result,
                      as_flat_file,
                      folder,
                      config.flat_device_status_csv_file_name,
                      file_type='csv')
    write_read_record(result,
                      as_flat_file,
                      folder,
                      config.flat_device_status_parquet_file_name,
                      file_type='parquet')
    print(f'Execution time: {timedelta(seconds=(time.time() - start_time))}')


if __name__ == "__main__":
    main()
