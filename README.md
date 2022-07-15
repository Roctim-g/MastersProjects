# AIT 580 Final Project:<br/>Understanding US Mass Shooting Data Last 50 Years
  Team 6: Roctim Gogoi, Nina Lin, Tony Sanchez
## Sytem Set Up

## Dataset Links
* EDA:
   - Mass shooting dataset: USMassShootings_RawData.csv 
* Machine Learning: 
   - Mass Shooting dataset: [USMassShootings07042022.csv](https://github.com/linxiuyun93/AIT580_Team6_Final-Project/blob/main/USMassShootings07042022.csv)
   - Gun Law Score Card: [GunLawScoreCard.csv](https://github.com/linxiuyun93/AIT580_Team6_Final-Project/blob/main/GunLawScoreCard.csv)
* NLP

## Exploratory Data Analysis
Executable script: [AIT580_T6_EDA_Project.py](https://github.com/linxiuyun93/AIT580_Team6_Final-Project/blob/main/EDA/AIT58_T6_EDA_Project.py) 

Python libraries: pandas, numpy, matplotlib.pyplot, plotly.express, seaborn

|Visualization result|
|---------------------|
|Some parts of the states have a high mass shooting rate as commpared to other states. From columns and geo charts, we saw that california is having the highest fatality and nevada has the highest injury rate as comapred to other states.|
|There is a correlation between year and the injury and fatalities. Year to fatality is 0.5, where year to injury 0.32 and year to total_victims 0.39.|
|There is a negative correlation between year and the age of the shooter which means as the year proceed from 1982, the age of the shooter who is involved in the shooting has come closer to younsters.|


## Machine Learning
Executable Script: [AIT580_Team6_Project_ML.py](https://github.com/linxiuyun93/AIT580_Team6_Final-Project/blob/main/AIT580_Team6_Project_ML.py)

Python libraries and functions:

|ML Model|Result|
|--------|------|
|Linear Regression(Gun law strength score vs. Fatalities)|R-Squared: |
|Linear Regression(Gun law strength score vs. Gun death rate)|R-Squared: |
|KKN Classification|Accuracy:  |
|Random Forest Classification|Accuracy: |
## NLP Analysis

