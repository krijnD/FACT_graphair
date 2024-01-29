



## Novel Dataset: full_with_ed.csv
This repository contributes a new combination of various datasets to get a complete overview
of congress member demographics for machine learning and data analysis purposes.

### Overview

- **Total**:
  - 254 congress members

- **Gender Counts**:
  - Male: 191
  - Female: 63

- **Chamber Counts**:
  - House: 198
  - Senate: 56

- **Party Counts**:
  - Democrat: 134
  - Republican: 120

- **Average Net Worth**: The average net worth across all individuals in the dataset is approximately $6,007,207.76.

### Column Descriptions


1. **Twitter**
   - Type: String
   - Description: The official Twitter handle of the individual.
   - Sample Values: ['SenatorBaldwin', 'SenJohnBarrasso', 'RoyBlunt', 'CoryBooker', 'SenSherrodBrown']
   
2. **First Name**
   - Type: String
   - Description: The first name of the individual.
   - Sample Values: ['Tammy', 'John', 'Roy', 'Cory', 'Sherrod']

3. **Last Name**
   - Type: String
   - Description: The last name of the individual.
   - Sample Values: ['Baldwin', 'Barrasso', 'Blunt', 'Booker', 'Brown']

4. **Party**
   - Type: String
   - Description: The political party affiliation of the individual, Democrat('D') or Republican ('R').
   - Sample Values: ['D', 'R']

5. **Bioguide ID**
   - Type: String
   - Description: A unique identifier assigned to each individual for biographical purposes.
   - Sample Values: ['B001230', 'B001261', 'B000575', 'B001288', 'B000944']

6. **Birthday Bio**
   - Type: String
   - Description: The birthdate of the individual.
   - Sample Values: ['1962-02-11', '1952-07-21', '1950-01-10', '1969-04-27', '1952-11-09']

7. **Gender Bio**
   - Type: String
   - Description: The gender of the individual.
   - Sample Values: ['F', 'M']

8. **Official Full Name**
   - Type: String
   - Description: The full official name of the individual.
   - Sample Values: ['Tammy Baldwin', 'John Barrasso', 'Roy Blunt', 'Cory A. Booker', 'Sherrod Brown']

9. **Religion Bio**
    - Type: String
    - Description: The religious affiliation of the individual.
    - Sample Values: ['Lutheran', 'Roman Catholic', 'Baptist', 'Presbyterian', 'Church of Christ']

10. **Minimum Net Worth**
    - Type: String
    - Description: The minimum estimated net worth of the individual.
    - Sample Values: ['$434,008', '$3,533,013', '$2,599,022', '$515,006', '-$163,988']

11. **Average Net Worth**
    - Type: String
    - Description: The average estimated net worth of the individual.
    - Sample Values: ['$1,147,003', '$8,339,006', '$5,704,510', '$807,503', '$263,005']

12. **Maximum Net Worth**
    - Type: String
    - Description: The maximum estimated net worth of the individual.
    - Sample Values: ['$1,859,998', '$13,145,000', '$8,809,999', '$1,100,000', '$689,998']

13. **Chamber**
    - Type: String
    - Description: The chamber of the legislature the individual serves in, such as the House of Representatives or the Senate.
    - Sample Values: ['Senate', 'House']

14. **State**
    - Type: String
    - Description: The state that the individual represents.
    - Sample Values: ['Wisconsin', 'Wyoming', 'Missouri', 'Alaska', 'Ohio']

15. **Highest Level of Education**
    - Type: String
    - Description: The highest level of education the individual has achieved.
    - Sample Values: ['J.D.', 'MD', 'MA', ' (MA, MPA)', 'BA']

16. **Highest Degree School**
    - Type: String
    - Description: The name of the school where the individual earned their highest degree.
    - Sample Values: ['University of Wisconsin–Madison', 'Georgetown University', 'Missouri State University', 'Willamette University', 'Ohio State University']

17. **Cleaned Average Net Worth**
    - Type: Float
    - Description: The cleaned numeric value of the individual's average net worth for

### Usage
This dataset can be used for analytical 
studies, such as assessing the correlation 
between net worth and political affiliations,
the diversity of educational backgrounds, or the
representation of states. It is also suitable for 
sociological studies into gender distribution and 
religious beliefs among political figures. We utilized it to test graph fairness while predicting the
average net worth of congress members based on their demographic factors. 

**PLEASE NOTE:** Using names, gender, age, and political affiliation
in this context will  introduce or amplify biases
based on gender, ethnicity, or other personal 
characteristics associated with demographic. Any 
analysis or model development should be approached 
with caution, ensuring transparency about the 
potential for bias and taking steps to mitigate 
these effects. This dataset was used to show systemic bias 
based on sensitive characteristics. We do not recommend utilizing this dataset to earnestly predict 
attributes such as net worth due to the bias that would entail and this dataset should 
not be employed in any decision-making based on these attributes. 



## Original Datasets

1. **congress_networth.csv**
   - Contains financial information about members of Congress, likely including their net worth, assets, liabilities, and other financial disclosures.
   - **Source**: 

2. **congress_twitter_handle_name.csv**
   - Provides a mapping between members of Congress and their Twitter handles, possibly including their names and other identifying information for social media analysis.
   - **Source**:

3. **legislators-current_biographic.csv**
   - Holds current biographical information about legislators, which might consist of their names, dates of birth, places of birth, tenure in office, and other personal details.
   - **Source**:

4. **legislators-social-media.csv**
   - Contains data on legislators' social media accounts across various platforms, not limited to Twitter but potentially including Facebook, YouTube, Instagram, and more.
   - **Source**:

5. **original_graph_usernames.txt**
   - Likely a text file with a list of usernames or identifiers for members of Congress, which may be used for network graphing or other forms of social analysis.
   - **Source**:

6. **US_Congress_Education_Data.csv**
   - Provides educational background information for members of Congress, such as degrees earned, institutions attended, fields of study, and graduation years.
   - **Source**:


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

## Warning
We cannot guarantee the validity or accuracy of this information. This data was pulled from a multitude of sources and merged together 
based on string parsing techniques. 

## Contribution
Contributions are welcome. If you wish to contribute, please fork the repository and submit a pull request.

## License
Specify your project's license here, which dictates how others can use the data and the code.
