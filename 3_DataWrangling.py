"""
Created on Fri Dec 23 11:18:30 2022
"""

# %% Set up

import pandas as pd
import numpy as np
import os

# Current directory
os.chdir('C:/Users/user/Desktop/Coursera/10_CapstoneDS')

# %% Download and save data

URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_1.csv'
df = pd.read_csv(URL)
df.to_csv('dataset_part_1_csv_fromcoursera.csv') # Save it just in case

# Check
df.head(10)

# Identify and calculate the percentage of the missing values in each attribute
df.isnull().sum()/df.shape[0]*100

# Types
df.dtypes

# %% TASK 1: Calculate the number of launches on each site

df.value_counts('LaunchSite')
'''
LaunchSite
CCAFS SLC 40    55
KSC LC 39A      22
VAFB SLC 4E     13
'''

# %% TASK 2: Calculate the number and occurrence of each orbit

df.value_counts('Orbit')
'''
Orbit
GTO      27
ISS      21
VLEO     14
PO        9
LEO       7
SSO       5
MEO       3
ES-L1     1
GEO       1
HEO       1
SO        1
'''

# %% TASK 3: Calculate the number and occurence of mission outcome per orbit type

landing_outcomes = df.value_counts('Outcome')
'''
Outcome
True ASDS      41  -- success
None None      19  -- fail
True RTLS      14  -- success
False ASDS      6  -- fail
True Ocean      5  -- success
False Ocean     2  -- fail
None ASDS       2  -- fail
False RTLS      1  -- fail

True Ocean means the mission outcome was successfully landed to a specific 
region of the ocean while False Ocean means the mission outcome was unsuccessfully 
landed to a specific region of the ocean. True RTLS means the mission outcome 
was successfully landed to a ground pad False RTLS means the mission outcome was 
unsuccessfully landed to a ground pad.True ASDS means the mission outcome was 
successfully landed to a drone ship False ASDS means the mission outcome was 
unsuccessfully landed to a drone ship. None ASDS and None None these represent 
a failure to land.
'''

# %% TASK 4: Create a landing outcome label from Outcome column

# Create set of outcomes where the second stage did not land successfully:
bad_outcomes=set(landing_outcomes.keys()[[1,3,5,6,7]])

# landing_class = 0 if bad_outcome. landing_class = 1 otherwise
landing_class = []
for key,value in df["Outcome"].items():
     if value in bad_outcomes:
        landing_class.append(0)
     else:
        landing_class.append(1)

df['Class']=landing_class

df[['Class']].head(8)

# Success rate
df["Class"].mean()
'''
0.6666
'''
# Export
df.to_csv("dataset_part_2.csv", index=False)
