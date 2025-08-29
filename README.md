# Investigating Diurnal Rhythm Disruption in T1D: Finding Patterns Using APS Data

## University of Bristol
### Owner: Ross Duncan
### Project Supervisor: Dr Zahraa Abdallah

This project provides the codebase used for data processing and analytics in support of Ross Duncan's Master's Thesis.

The thesis can be found in reports/Thesis_Final_Version.pdf

The project contents follows the following structure:

    - root                          : Project root directory  
        - data                      : Data directory will be where data does pre and post processing
        - notebooks                 : This holds analysis used in the project
        - report                    : Holds the thesis report
        - src                       : Source content of the project, holds files with classes uses
            - data_processing       : Source used in data processing, from raw dataset
            - scripts               : Pipelines used to extract and process data
        - tests                     : Unit tests

Once forked and the environment built using the requirements.txt file, the private.yaml file needs changing to the data source of the raw data, and options set, to begin.
