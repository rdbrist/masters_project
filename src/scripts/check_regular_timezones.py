from pathlib import Path

from src.data_processing.read import  (read_all_device_status,
                                       read_all_profile_timezones,
                                       convert_timezone_to_utc_offset,
                                       convert_all_timezones_to_utc_offset)
from src.configurations import Configuration

if __name__ == "__main__":
    config = Configuration()
    config.data_folder = Path("/data/raw")
    profile_timezones = read_all_profile_timezones(config)
    profile_offsets = convert_all_timezones_to_utc_offset(profile_timezones)

    read_device_records = read_all_device_status(config)
    device_timezones = {}
    for rr in read_device_records:
        device_timezones[rr.zip_id] = rr.timezones
    device_offsets = convert_all_timezones_to_utc_offset(device_timezones)

    def dict_diff(dict1, dict2):
        diff = {}
        all_keys = set(dict1) | set(dict2)
        for key in all_keys:
            if dict1.get(key) != dict2.get(key):
                diff[key] = (dict1.get(key), dict2.get(key))
                print(f"Key: {key}, Profile: {dict1.get(key)}, Device: {dict2.get(key)}")
        return diff

    print_diff = dict_diff(profile_offsets, device_offsets)
    print(f"From {len(profile_timezones)} profiles, and {len(read_device_records)} device status individuals:")
    print(f"Number of differences: {len(print_diff)}")
