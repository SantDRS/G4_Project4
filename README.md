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
