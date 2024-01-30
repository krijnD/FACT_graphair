

## Congress Benchmark Data 


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
Found in `original_datafiles`the datasets and files used to create out benchmark dataset.

1. **congress_networth.csv**
   - Contains financial information about members of Congress, specifically their net worth estimations.
   - **Please Note**: This is from 2018 and is not up-to-date information.
   - **Source**: https://www.opensecrets.org/personal-finances/top-net-worth 

2. **congress_twitter_handle_name.csv**
   - Provides a mapping between members of Congress and their Twitter handles, including their names and other identifying information for social media analysis.
   - **Source**: https://pressgallery.house.gov/member-data/members-official-twitter-handles 
   - **Source**: https://www.kaggle.com/datasets/thedevastator/us-congress-legislators-historical-data

3. **legislators-current_biographic.csv**
   - Holds current biographical information about legislators, which consist of their names, dates of birth, gender, ids and other personal details.
   - **Source**: https://www.kaggle.com/datasets/thedevastator/us-congress-legislators-historical-data
   - **Source**: https://en.wikipedia.org/wiki/Religious_affiliation_in_the_United_States_House_of_Representatives

4. **legislators-social-media.csv**
   - Contains data on legislators' social media accounts across various platforms, not limited to Twitter but including Facebook, YouTube, Instagram, and more.
   - **Source**: https://www.kaggle.com/datasets/thedevastator/us-congress-legislators-historical-data

5. **original_graph_usernames.txt**
   - Text file with a list of twitter (I guess X now) handles for members of Congress, which are used for network graphing or other forms of social analysis.
   - **Source**: https://github.com/gsprint23/CongressionalTwitterNetwork/tree/v1?tab=readme-ov-file 

6. **US_Congress_Education_Data.csv**
   - Provides educational background information for members of Congress, such as degrees earned, institutions attended, fields of study, and graduation years.
   - **Source**: https://www.kaggle.com/datasets/philmohun/complete-education-details-116th-us-congress 

7. **Congressional Twitter Network**
    - 
    - **Source Repository**:
    - **Source Paper**:
    - For more information on individual files please check the original data repository.

## Processing non-Graph Information
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
- Biographic Information

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

### Dataset: full_with_ed.csv
This repository contributes a new combination of various datasets to get a complete overview
of congress member demographics for machine learning and data analysis purposes.

#### Overview

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



### Feature Dataset: encoded_data.csv
For utilization in our project (and generally)  we transformed 
all the categorical variables gathered in `full_with_ed.csv` into various
numerical or binary values, most are one-hot encoded. In 

#### Overview
Not in any specific order.

1. **Twitter** 
   - Type: String
   - Description: The official Twitter handle of the individual.
   - Sample Values: ['SenatorBaldwin', 'SenJohnBarrasso', 'RoyBlunt', 'CoryBooker', 'SenSherrodBrown']

2. **Class Net Worth**
    - Type: Integer
    - Description: Based on the mean net worth value in U.S. Dollars found in `Cleaned Average Networth` in `full_with_ed.csv`. The average of these values for these congress people were found and separated into 3 classes: 0 if their net worth is on or around the average (10% buffer), 1 if it is above the average, -1 if it is below the average.
    - Sample Values: [1, 0, -1]


3. **feat_party**
    - Type: Binary Integer
    - Description: If the party is 'D' for Democrat then it is encoded as 1, if the party is Republican ('R') then value is 0.
    - Sample Values: [0,1]

4. **age**
    - Type: Integer
    - Description: Age of congress member calculated with the birthday in column `birthday_bio`
    - Sample Values: [63, 75]
    - **Please Note**: Age is time-dependent. This is the congress members age as of 28.01.2024

5. **gender_feat**
    - Type: Binary Integer
    - Description: Based on the `gender_bio` column. Value is 0 if 'M' indicating Male, otherwise it is 1, in this particular case 1 only represents 'F' Female as those were the gender identifies given. Thus, please note in theory 1 can also represent other non-Male genders on the gender spectrum.
    - Sample Values: [0,1]

6. **Education Degrees One-Hot Encoded**
    - Column Names: ['AA', 'AAS', 'AB', 'ALB', 'BA', 'BS', 'DDS', 'DVM',
       'EdD', 'JD', 'JD MBA', 'LLM', 'MA', 'MA PhD', 'MBA', 'MD', 'MDiv',
       'MEd', 'MFA', 'MPA', 'MPP', 'MPhil', 'MS', 'MSS', 'MSW', 'PhD']
    - Value Type: Binary Integer
    - Description: Each column has a binary value 1 if this is the highest congress member educational degree and 0 if not. Some members have more than one.
    - Sample Values: [0,1]

7. **chamber_feat**
    - Type: Binary Integer
    - Description: Based on the `Chamber` column. Value is 1 if the congress person is in the House and 0 if they are in the Senate.
    - Sample Values: [0,1]

8. **University Information One-Hot Encoded**
    - Column Names:['Tech School Grad', 'Ivy League Grad',
       'State University Grad', 'Community College Grad', 'Drop Out']
    - Type: Binary Integer
    - Description: Due to the large amount of unique education places of congress members we sorted universities from the `Highest Degree School` column into a few categories. Specifically, we used a 1 if their university could be classified under the umbrella term we established. We focused on Technology Schools, Ivy League (https://en.wikipedia.org/wiki/Ivy_League), State School, and Community College.
    - Sample Values: [0,1]
    - **Please Note**: The Tech, State, and Community College labels were assigned by keywords in a University name such as 'State', 'Technology' and 'University of'. Thus, they cannot be guaranteed to be completely correct university classifications. The categories are not mutually exclusive.
 
9. **first_name_vector, last_name_vector**
    - Type: pre-trained Word Vector
    - Description: Congress member first and last name were separately encoded as word vectors using gensim "glove-wiki-gigaword-100" GloVe model.
    - **Please Note**: Not used in our project due to parsing issues and not relevant

10. **Religious Affiliation Grouped**
    - Column Names: ['Christian', 'Jewish', 'Other', 'Unknown']
    - Type: Binary Integer
    - Description: The religions found in the `religion_bio` column of `full_with_ed.csv` have been grouped under a few umbrella terms and made into one-hot encodings. If the congress member has a religious affiliation to that group 1, else 0.
    - Sample Values: [0,1]
    - **Please Note**: We did not find a lot of parsable data for this and thus a lot of them are unknown. They are grouped under religions based on Wikipedia answers, we do not know much about the different subsets of Christianity.

11. **State** 
    - Column Names: https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States
    - Type: Binary Integer
    - Description: One-hot encoding in each column for all the States in the U.S.A.
    - Sample Values: [0,1]

12. **Religion Sensitivity**
    - Type: Binary Integer
    - Description: This column was created to examine religion as a sensitive attributes. This encoding was applied to the grouped religious affiliation column. Minority religions in Congress (in this case for this dataset that is Jewish or Other) were encoded with a 1. Majority religious groups which were Unknown and Christianity have value 0. 

13. **numeric_id**
    - Type: Integer
    - Description: A numeric identifiers corresponding to member twitter handles. It is used in the Graph node data.
    - Sample Values: [4, 8, 12]

## Processing Graph Dataset
To make the `data_merging_and_processing/original_datafiles/CongressionalTwitterNetwork` into a format parsable 
by `dig\fairgraph` we applied some processing methodology

### connections_weights.json
`connections_weights.json` is a subset of the original congressional data from the `Congress Twitter Network`. 
It contains the nodes for the relevant members based on their presence in the biographic data file `full_with_ed.csv`.
The file is created with `data_merging_and_processing/graph_subset.py`, a rough processing Python script to isolate the relevant members via Twitter handles.


### cng_relationship.txt
This file contains the node relationships relevant for our subset in a format compatible with Graphair.
Each line is structured as source_id\ttarget_id\n, indicating the relationships between entities using numeric IDs.
It was created using `data_merging_and_processing/graph_relationship.py` and one can find the username and numeric id mapping in the Python file `data_merging_and_processing/get_mapping.py`

## References and Acknowledgements 


## Warning
We cannot guarantee the validity or accuracy of this information. This data was pulled from a multitude of sources and merged together 
based on string parsing techniques. 

## Contribution
Contributions are welcome. If you wish to contribute, please fork the repository and submit a pull request.

## License
Specify your project's license here, which dictates how others can use the data and the code.
