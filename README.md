# Smart Travel App
Group 4, Project 4
Team: Dominique Villarreal, Enrique Garcia, Jose Santos
Project Due Date: June 12, 2023

# Purpose:
The objective of this project is to develop a sophisticated machine learning solution that you can curate personalized itineraries for your trips. By harnessing the power of advanced algorithms and data analysis, our system will assist you in planning and optimizing your travel experiences.

Through a seamless integration of artificial intelligence and travel data, we will pick out the city that best suits you based on your choices of activities. After your city is chosen we will provide you with an enormous amount of activities to do. 

By utilizing cutting-edge technologies and comprehensive data sources, our aim is to provide you with a highly planned and optimized travel plan. The system will consider a wide range of activities to choose from examples.(skiing, bike riding, bars, night life, beaches, etc.).

With our machine learning-powered category generator, you can unlock the potential of a personalized travel experience. Say goodbye to the headaches and struggles of planning and let our intelligent system take the lead, ensuring that every moment of your trip is optimized, memorable, and tailored to your unique preferences. You will never run out of ideas to do in the city. 

# Steps:

## G4_P4_Notebook1_DataPull
In this Notebook we explore the Yelp API to find the data we believe would best support our project. In doing so we found that pulling based off of “parent_aliases” would give us the best data. Once we chose the parent aliases that we believed would best fit the cities we selected, we began making API calls by city.

### Prerequisites
Create an API key using the following link, https://www.yelp.com/login?return_url=/developers/v3/manage_app 
Make sure you have the following dependencies installed:
- import json
- import pandas as pd
- import pprint as pp
- import requests
- import time

### Usage
**Disclaimer: You must insert your created API key everywhere in the code after “Authorization”, replacing the API key already in that line.
The API limits each call to 50 rows of data, so we use a loop to call up to 1000 rows of data (when available) using a rest between calls to avoid being locked out. Even then, there is a maximum pull limit of 6 a day. So data was pulled over a span of 3 days.  **

1. Set up your environment and install the required dependencies.
2. Make an API call that calls in categories with its respective parent alias
3. Make an API call that calls in parent aliases
4. Use a loop to call in up to a thousand rows of data for each parent alias for each city (Denver, Miami, New York) for the following aliases: 
- Active
- Arts
- Bars
- Breweries
- Festivals
- Fitness
- Food
- Museums
- Night Life
- Restaurants
- Tours
- Wineries
- Zoo’s
5. Within each loop above, after each call, the loop appends the call data to a json file and is written out to the folder and printed on the screen. The user will note that the print command will only print the last dictionary. 
6. Proceed to G4_P4_Notebook2_DataCleaning.ipynb

## G4_P4_Notebook2_DataCleaning
In this notebook, we work on bringing in the previously made json files and combining them into a single data frame (by city). Where we then cleaned/ manipulated the data frame to best represent the data for our needs. 

### Prerequisites 
Make sure you have the following dependencies installed:
- import json
- import pandas as pd
- import pprint as pp
- import requests
- import time
### Usage

1. Set up your environment and install the required dependencies.
2. Create a path to the folder housing the json files for the respective city. Make an empty list to store the data frames that are going to be built in the loop. In the loop we iterate over the files in the folder, read the file contents, parse the file contents into a json object, from the json object we create and append the data frames to the list we made before the loop. We then concatenate the list of the data frames into a single data frame. 
3. Once we have a single data frame, the categories column contains a dictionary within a list, so we read the dictionary for “title” in categories and assign it to category names. We then delete the original categories column and assign category names to a new categories column. 
4. In the next section we delete the unnecessary columns from the data frame that contain irrelevant data. 
5. We then add a column named ‘City’, for the respective city that we are working on.
6. Once the city column is added we reorder the columns in a way that best represents the data.
7. Now that the data frame is structured we can export it to a folder as a csv. To do that we create a file path to the desired end location, convert the data frame into a CSV and export it using the path created.
8. Rinse and repeat steps 1-6 for Miami and New York.

## G4_P4_Notebook3_DataProcessing
In this notebook we bring in the previously created data frames for each city and merge them into one single data frame.

### Prerequisites
Make sure you have the following dependencies installed:
- import json
- import os
- import pandas as pd
- import pprint as pp
- import requests
- import time

### Usage
1. Set up your environment and install the required dependencies.
2. In the first 3 cells after setting up your dependencies, we create a path to the folder housing the csv files just created and read them in. 
3. Using the .info() function on each city’s respective dataframe, we make sure the dtypes match up for each column in each city’s df.
4. We then use the .explode(), and .unique() functions to find the unique categories within each city. 
5. We then use the .concat() function to merge all 3 data frames into one data frame named merged_df
6. Once again, we use the .explode(), and .unique() functions to find the unique categories within the merged data frame.
7. Now that the data frames have been merged we can export merged_df to a folder as a csv. To do that we create a file path to the desired end location, convert the data frame into a CSV and export it using the path created.

## G4_P4_Notebook4_MLvAllData
In this Notebook we use a Random Forest Classifier to predict cities and their corresponding activities based on a randomly selected set of categories. The code generates a random sample of 25 categories, predicts the city using the trained classifier, and retrieves the top activities for each category in the predicted city. Each time you run the code it will choose a different random set of categories to make a unique prediction. 

### Prerequisites
Make sure you have the following dependencies installed:
- import json
- import os
- import pandas as pd
- import pprint as pp
- import random
- import requests
- import time
- import warnings
- warnings.simplefilter("ignore")
- from sklearn.cluster import KMeans
- from sklearn.preprocessing import LabelEncoder
- from sklearn.preprocessing import MinMaxScaler
- from sklearn.preprocessing import StandardScaler
- from sklearn.ensemble import RandomForestClassifier
- from sklearn.model_selection import train_test_split

### Usage
1. Set up your environment and install the required dependencies.
2. Import merged_df.csv from notebook 3
3. Remove columns location and coordinates that we don't need for prediction process & rearrange the columns for easier viewing
4. Check the .info() for the merged_df 
5. Create city_encoded_data df by copying merged_df
6. Explode the 'categories' column to create a row for each category listed in categories column
7. Create label_encoder to encode categories and use in machine learning &   Apply label encoding to the 'categories' column
8. Write out city_encoded_data to csv
9. Create city_category_predict_df to avoid breaking city_encoded_data
10. Random Forest ML Model - All Data - Model Creation
    - Select X & y and reshape df
    - Split the data into training and test sets
    - Create and train the Random Forest Classifier
11. Train & Test Our Model
    - Fit the training data to the model
    - Review the accuracy of the model on the training data
        - Accuracy on training data of : 0.5381600249950294
    - Fit the test data to the model
    - Review the accuracy of the model on the test data
        - Accuracy on training data of : 0.5457850488525335
12. Predict the City
    - Generate a random sample of 25 categories. This process is mean to mimic anyone one persons 25 category selection
    - Create df for random sample
    - Assign the readable category names based on random sample of the 25 randomly selected categories above
    - Print the readable categories being used to predict the city
    - Predict the city based on the random group of 25 categories
13. List Experiences Within Predicted City
    - List top 3 experiences for each category within predicted city 
    - Set to keep track of predicted cities and their activities
    - Set to keep track of unique experiences
    - Get the top activities for each sample category
        - Convert category back to its name
        - Create a new activities_data list for each category iteration
        - Create for loop to iterrow over top_activities
            - Create a unique experience key
            - Check if the experience is unique
            - Create if not in statement to select only unique experiences
                - Add the experience to the set of unique experiences
                - Append the predicted activity to the list
        - Append the activities_data to the predicted_data list for each category
    - Create a DataFrame from the predicted data
    - Print the final predicted data DataFrame
14. Proceed to G4_P4_Notebook5_MLvSelectedData to run a model on unique categories only
15. Output Predictions to csv - The final predicted data DataFrame can be saved as a CSV file using the following pathname: `../G4_Project4/Resources/Insights/predicted_df.csv`.
