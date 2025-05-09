import pandas as pd

from src.cob_analysis import Cob

# set as_flat_file to True if you want to save one big flat bg csv or set it to false if you want a bg_csv per id
def main():
    cob = Cob()
    cob.read_processed_data()
    c = cob.processed_dataset
    print(c.head())
    # peaks = c[c['peak'] == 1].index.get_level_values('datetime').hour
    # peak_counts = pd.Series(peaks).value_counts()
    # print(peak_counts)

if __name__ == "__main__":
    main()