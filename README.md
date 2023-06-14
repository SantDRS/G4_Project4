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
The API limits each call to 50 rows of data, so we use a loop to call up to 1000 rows of data (when available) using a rest between calls to avoid being locked out. Even then, there is a maximum pull limit of 6 a day. So data was pulled over a span of 3 days. Keep in mind if you run all these cells at the same time you run the risk of getting kicked out of the api.  **

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
In this notebook, we work on bringing in the previously made json files and combining them into a single data frame (by city). Where we then manually cleaned/ manipulated the data frame to best represent the data for our needs. 

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

## G4_P4_Notebook4_MLvAllData-Enhanced
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

## G4_P4_Notebook5_MLvSelectedData
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
1. Load Environment
    - Set up your environment and install the required dependencies.
    - Import merged_df.csv from notebook 3
    - Remove columns location and coordinates that we don't need for prediction process & rearrange the columns for easier viewing
    -Check the .info() for the merged_df 
2. Process Data
    - Create city_encoded_data df by copying merged_df
    - Explode the 'categories' column to create a row for each category listed in categories column
    - Create label_encoder to encode categories and use in machine learning & Apply label encoding to the 'categories' column
    - Write out city_encoded_data to csv
    - Begin filtering city_encoded_data df by city to find the unique categories
        - Denver
        - Filter the DataFrame for rows with the city "Denver"
        - Get the unique categories for Denver
        - Filter the DataFrame for rows with cities other than "Denver"
        - Get the unique categories for the other cities
        - Get the categories that are unique to Denver
        - Create a DataFrame with the unique categories for Denver
        - Add city column
        - Rename unique categories column
        - Drop Unique Categories for New York
        - Print the DataFrame
    - View denver_unique_categories_df info
    - Repeat Steps 9 & 10 for cities: Miami & New York
    - Concatenate the DataFrames
    - View combined_unique_df info
    - Export combined unique csv
    - >>>>> The combined_unique_categories.csv was manually scrubbed to keep only the categories that were not so general and more obviously specific to each city. This was done to make it possible for our computers to process all the data. - continue to next step 
    - Create a filtered df using the selected df as a filter to the categories 
    - Create the city_category_selected_predict_df
    - Output Data
        - Write out city_encoded_data to csv
3. Random Forest ML Model - All Data - 
    - Model Creation
        - Select X & y and reshape df
        - Split the data into training and test sets
        - Create and train the Random Forest Classifier
    - Train & Test Our Model
        - Fit the training data to the model
        - Review the accuracy of the model on the training data
            - Accuracy on training data of : 0.5381600249950294
        - Fit the test data to the model
        - Review the accuracy of the model on the test data
            - Accuracy on training data of : 0.5457850488525335
    - Predict the City
        - Generate a random sample of 25 categories. This process is mean to mimic anyone one persons 25 category selection
        - Create df for random sample
        - Assign the readable category names based on random sample of the 25 randomly selected categories above
        - Print the readable categories being used to predict the city
        - Predict the city based on the random group of 25 categories
    - List Experiences Within Predicted City
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
4. Output Predictions to csv - The final predicted data DataFrame can be saved as a CSV file using the following pathname: `../G4_Project4/Resources/Insights/predicted_df.csv`.
5. The End of Notebook Model Prep & Processing. Final notebook G4_P4_Notebook6_VisualizationData is only for preparing data formats for visualizations.

## G4_P4_Notebook6_VisualizationData
In this notebook we restructure the data frames to bring back in location and coordinate data to bring in visualizations to create in tableau. 
### Prerequisites
Make sure you have the following dependencies installed:
-import json
-import pandas as pd
-import pprint as pp

### Usage
1. Set up your environment and install the required dependencies.
2. Import merged_df that was created in notebook 
3. Process and organize the merged_df. You will insert a list using, city, name, rating, review count, categories, location, and coordinates.
4. Used merged_df.info to show the dtypes
5. Make a copy of the merged data frame and name it vr_data
6. Explode the categories column so that each category is in its own column
7. Convert the string coordinates to dictionaries 
8. Split the coordinates column into latitude and longitude. Coordinates will now have both a latitude and longitude row.

![Screen Shot 2023-06-11 at 2 30 39 PM](https://github.com/SantDRS/G4_Project4/assets/120337088/2c121b04-3e19-4748-b1cf-53741fbd825d)

9.Drop the coordinates column now that it is unnecessary to keep.
![Screen Shot 2023-06-11 at 2 28 15 PM](https://github.com/SantDRS/G4_Project4/assets/120337088/2e7d9774-6f4f-4758-ade7-27c4892960e1)

10. Save vr_data to a csv
11. Pull in selected_unique_categories.csv. Name the data frame ‘selected_unique_cat_df’
12. Use vr_data to filter out the unique categories.

![Screen Shot 2023-06-11 at 2 31 39 PM](https://github.com/SantDRS/G4_Project4/assets/120337088/440c030b-037c-44ee-8037-488e491eea83)
13. Save to csv and move to tableau. 
14. Once in tableau add in vr_data as your datasource 
15. Put categories in both columns and rows change columns category to attribute then change rows category to count
16. For the next visual you will use count categories in column and for rows you will insert city and categories as descending. 
17. For total categories x city you will put city as columns and categories as rows make sure it is measured in count distinct.
18. In selected categories x city you will put city from vr_data_selected.csv and categories from vr_data_selected as measured count distinct.
19. For selected categories x type you want to use ve_data_selected. Add categories as count in column and for rows you want to add city, and categories. 
20. Denver all you will add in columns as longitude and rows as latitude. You will filter out Denver. In tooltip you will add in name, review count, location split 8-1, and location split 8-4 along with raining. You will want to make sure all are under attribute. 
21. Repeat the same steps as number 20, but change the filter as city fro vr_data_selected. 
22. Duplicate Denver all twice and filter out to Miami and the next New York.
23. Duplicate Denver selected twice and filter as well to Miami and the next to New York. 

# Tableau Visualizations 
https://public.tableau.com/app/profile/dominique.villarreal/viz/Project4full/categoriesxtype?publish=yes


#visulizations
![Picture1 tab](https://github.com/SantDRS/G4_Project4/assets/120337088/1dffa59d-2e87-4df7-ba57-0276ad4442a2)
![pic2 tab](https://github.com/SantDRS/G4_Project4/assets/120337088/52bcca0e-7fb7-4fcc-a7f5-31bfda2aa5fe)
![pic3 tab](https://github.com/SantDRS/G4_Project4/assets/120337088/251918d1-c490-444a-a537-784838badfff)

![pic4 tab](https://github.com/SantDRS/G4_Project4/assets/120337088/59a574f7-73b3-431f-ba55-057df42e89bc)
![pic5 tab](https://github.com/SantDRS/G4_Project4/assets/120337088/74594d01-b5d6-482d-959e-ac6d7e1988f8)

![pic6 tab](https://github.com/SantDRS/G4_Project4/assets/120337088/7e757661-026a-4bfb-a85e-bedd4f75a0a6)

![pic7 tab](https://github.com/SantDRS/G4_Project4/assets/120337088/22ac6c93-7938-423c-9f7a-813a18f4a5f9)
![pic8 tab](https://github.com/SantDRS/G4_Project4/assets/120337088/bb647175-1bec-4a7d-acd8-012a858b6ace)
![pic9 tab](https://github.com/SantDRS/G4_Project4/assets/120337088/ae1f6719-0cc4-475d-8630-4080e01bfe9d)
![pic10 tab](https://github.com/SantDRS/G4_Project4/assets/120337088/b9929d0c-5296-456a-ad4b-b52f5f18ab1d)
![pic11 tab](https://github.com/SantDRS/G4_Project4/assets/120337088/cb4422be-5f41-4067-a81c-a368ea5c03bf)
![pic12 tab](https://github.com/SantDRS/G4_Project4/assets/120337088/b29f4779-a27a-4595-afeb-93666824a356)
![pic13 tab](https://github.com/SantDRS/G4_Project4/assets/120337088/5b9cf73f-53c0-415c-a7c4-7cf7390f7f96)

#Conclusion: 
When conducting data analysis across all categories, the accuracy achieved is  53.82 percent train 54.58 percent test. However, by employing a more focused approach and hand-selecting unique categories, the accuracy significantly improves to  100 percent for train and test. This highlights the importance of unique category filtering in optimizing accuracy and obtaining reliable results. By refining the analysis process and narrowing down the categories, we can unlock valuable insights with a higher degree of precision, empowering decision-making and enabling actionable outcomes. 


