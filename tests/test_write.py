import pytest
import pandas as pd
from src.configurations import Configuration
from src.data_processing.read import read_all_bg, read_all_device_status
from src.data_processing.write import write_read_record
from tests.test_read import input_file


@pytest.mark.parametrize("read_function", [read_all_bg, read_all_device_status])
def test_write_read_record(input_file, tmp_path, read_function):
    # Setup configuration
    config = Configuration()
    config.data_dir = str(input_file.parent)  # Set the data directory

    # Generate records using the read function
    records = read_function(config)

    # Ensure records are valid
    assert isinstance(records, list), "Records should be a list"
    assert len(records) > 0, "Records list should not be empty"

    # Test writing as a flat file
    flat_folder = tmp_path / "flat"
    flat_folder.mkdir()
    flat_file_name = "test_flat.csv"
    write_read_record(records, as_flat_file=True, folder=flat_folder, file_name=flat_file_name)
    flat_file_path = flat_folder / flat_file_name
    assert flat_file_path.exists(), "Flat file should be created"

    # Validate flat file content
    flat_df = pd.read_csv(flat_file_path)
    assert not flat_df.empty, "Flat file should not be empty"
    assert "time" in flat_df.columns or "created_at" in flat_df.columns, "Timestamp column should exist"

    # Test writing per ID
    per_id_folder = tmp_path / "per_id"
    per_id_folder.mkdir()
    per_id_file_name = "test_per_id.csv"
    write_read_record(records, as_flat_file=False, folder=per_id_folder, file_name=per_id_file_name)

    # Validate per ID files
    for record in records:
        per_id_file_path = per_id_folder / record.zip_id / per_id_file_name
        assert per_id_file_path.exists(), f"File for ID {record.zip_id} should be created"
        per_id_df = pd.read_csv(per_id_file_path)
        assert not per_id_df.empty, f"File for ID {record.zip_id} should not be empty"
        assert "time" in per_id_df.columns or "created_at" in per_id_df.columns, "Timestamp column should exist"