



## Novel Dataset: full_with_ed.csv


## Original Datasets


## Processing
To see a rough version of the code used to take the original dataset csv and
merge them into a full dataset please see `data_merging_and_processing/data.ipynb`. 
Here one can find a jupyter notebook (made and utilized in Google Colab) that processed the original data files and merged them
into `full_with_ed.csv` and the process of encoding features from `full_with_ed.csv` into numerical features in `encoded_data.csv`

### Description
This project involves extensive data processing and manipulation for a dataset related to [specific domain, e.g., U.S. Congress data]. The data encompasses various aspects such as education details, net worth, social media presence, and other personal attributes of individuals associated with the domain.

### Data Sources
The data is sourced from multiple CSV files, which are processed and merged to form a comprehensive dataset. The sources include:

- Education data
- Net worth information
- Social media profiles
- Current status or position data

### Data Processing Steps
The Jupyter notebook `data.ipynb` contains all the data processing logic, which includes:

1. **Data Loading**: CSV files are loaded into pandas DataFrames for manipulation.
2. **Data Cleaning**: Various cleaning operations are performed, such as:
    - Standardizing string fields (e.g., Twitter handles).
    - Dropping unnecessary rows and columns.
    - Handling missing values by filtering or filling them.
3. **Data Merging**: Data from different sources are combined based on common attributes.
4. **Data Deduplication**: Duplicate entries are identified and removed to ensure data integrity.
5. **Data Extraction**: Regular expressions are used to extract structured data from text fields.
6. **Data Transformation**: The data undergoes transformations to ensure it is in a usable format for analysis, including:
    - Merging related columns.
    - Splitting columns into more granular parts.
    - Aggregating information where necessary.
7. **Data Storage**: The processed data is saved into a clean CSV file for downstream analysis or modeling.

### Output
The final output is a cleaned and consolidated dataset stored as `encoded_data.csv` which can be utilized for further analysis or modeling tasks.

### Dependencies
- Python 3.x
- Pandas
- NumPy
- Matplotlib (optional, for any data visualization if included)


## Feature Dataset: encoded_data.csv


## References and Acknowledgements 

## Contribution
Contributions are welcome. If you wish to contribute, please fork the repository and submit a pull request.

## License
Specify your project's license here, which dictates how others can use the data and the code.
