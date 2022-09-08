
#Hello, This analysis I will be using U.S. Major League Soccer Salaries
# link:https://www.kaggle.com/datasets/crawford/us-major-league-soccer-salaries
# This analysis task has allowed me to have a good foudnation in Numpy and Python as wella s had on hands experience
# With my certificate course I have both improved my Python skills in Data analysis as well as frameworks
#But also solidified my knowledge in Python foundations

#Let's improt the Numpy and Pandas

import numpy as np
import pandas as pd 

# Reading the dataset 

soccer = pd.read_csv("Portfolio/mls-salaries-2017.csv")
soccer


#Reading the first 10 rows

soccer.head(n = 10)

#Calculating how many rows are in dataset

len(soccer)

#Average Salary

soccer["base_salary"].mean()

#Higest salary detected in USD 

soccer["base_salary"].max()

#Then, I have detected the last name of the soccer player 
# who got the highest compensation 

soccer[soccer["guaranteed_compensation"].max() == soccer["guaranteed_compensation"]]

soccer[soccer["guaranteed_compensation"].max() == soccer["guaranteed_compensation"]]["last_name"]

#I grouped people based on their position and computed their mean salary

soccer.groupby("position").mean()

#Detected how many types of different positions in the dataset

soccer["position"].nunique()

#I then found how many players in each position

soccer["position"].value_counts()

#Detected number of players in different teams

soccer["club"].value_counts()



