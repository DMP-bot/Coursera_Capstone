"""
Created on Fri Dec 23 18:30:55 2022
"""

# %% Set up

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Current directory
os.chdir('C:/Users/user/Desktop/Coursera/10_CapstoneDS')

# %% Load data

URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
df = pd.read_csv(URL)
df.to_csv('dataset_part_2_csv') # Save it just in case

# %% Task 0: Provided by coursera
    
# Current directory
os.chdir('C:/Users/user/Desktop/Coursera/10_CapstoneDS/Figures')

# Flight number & Payload mass (&outcome)
sns.catplot(y="PayloadMass", x="FlightNumber", hue="Class", data=df, aspect = 5)
plt.xlabel("Flight Number",fontsize=20)
plt.ylabel("Pay load Mass (kg)",fontsize=20)
plt.savefig('Task0.png')

# %% TASK 1: Visualize the relationship between Flight Number and Launch Site

# Use the function catplot to plot FlightNumber vs LaunchSite, 
# set the parameter x parameter to FlightNumber,set the y to Launch Site and 
# set the parameter hue to 'class'
sns.catplot(data=df, x="FlightNumber", y="LaunchSite", hue='Class')
plt.xlabel("Flight Number",fontsize=20)
plt.ylabel("Launch Site",fontsize=20)
plt.savefig('Task1.png')

# %% TASK 2: Visualize the relationship between Payload and Launch Site

# Plot a scatter point chart with x axis to be Pay Load Mass (kg) and y axis 
# to be the launch site, and hue to be the class value
sns.catplot(data=df, x="PayloadMass", y="LaunchSite", hue='Class')
plt.xlabel("Pay Load Mass (kg)",fontsize=20)
plt.ylabel("Launch Site",fontsize=20)
plt.savefig('Task2.png')

# %% TASK  3: Visualize the relationship between success rate of each orbit type

# Create a bar chart for the sucess rate of each orbit
# HINT use groupby method on Orbit column and get the mean of Class column
dfgroup = df.groupby(by=['Orbit']).mean()
dfgroup = dfgroup[['Class']]
dfgroup['Orbit'] = dfgroup.index


# %% TASK  4: Visualize the relationship between FlightNumber and Orbit type

# Plot a scatter point chart with x axis to be FlightNumber and y axis to be 
# the Orbit, and hue to be the class value
sns.catplot(data=df, x="FlightNumber", y="Orbit", hue='Class')
plt.xlabel("Flight Number",fontsize=20)
plt.ylabel("Orbit",fontsize=20)
plt.savefig('Task4.png')


# %% TASK  5: Visualize the relationship between Payload and Orbit type

# Plot a scatter point chart with x axis to be Payload and y axis to be the 
# Orbit, and hue to be the class value
sns.catplot(data=df, x="PayloadMass", y="Orbit", hue='Class')
plt.xlabel("Payload Mass",fontsize=20)
plt.ylabel("Orbit",fontsize=20)
plt.savefig('Task5.png')

# %% TASK  6: Visualize the launch success yearly trend

# A function to Extract years from the date 
year=[]
def Extract_year():
    for i in df["Date"]:
        year.append(i.split("-")[0])
    return year
Extract_year()
df['Date'] = year
df.head()
    
# Plot a line chart with x axis to be the extracted year and y axis to be the success rate
dfgroup = df.groupby(by=['Date']).mean()
dfgroup = dfgroup[['Class']]
dfgroup['Date'] = dfgroup.index

sns.lineplot(data=dfgroup, x='Date', y='Class',color='b')
plt.xlabel("Year",fontsize=10)
plt.ylabel("Success rate (%)",fontsize=10)
plt.savefig('Task6.png')

## Features Engineering
features = df[['FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights', 
               'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial']]
features.head()

# %% TASK  7: Create dummy variables to categorical columns

# Use the function get_dummies and features dataframe to apply OneHotEncoder to
# the column Orbits, LaunchSite, LandingPad, and Serial. Assign the value to the
# variable features_one_hot, display the results using the method head. Your 
# result dataframe must include all features including the encoded ones.

features_one_hot = pd.get_dummies(features[['Orbit', 'LaunchSite', 'LandingPad','Serial']])
features_one_hot.head()

# %% TASK  8: Cast all numeric columns to `float64`

# HINT: use astype function
features_one_hot.astype('float64')

# Current directory
os.chdir('C:/Users/user/Desktop/Coursera/10_CapstoneDS')
features_one_hot.to_csv('dataset_part_3.csv', index=False)
