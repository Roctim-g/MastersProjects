# AIT 580 Final Project:<br/>Understanding US Mass Shooting Data Last 50 Years
  Team 6: Roctim Gogoi, Nina Lin, Tony Sanchez
## Sytem Set Up

## Dataset Links
* EDA:
   - Mass shooting dataset: USMassShootings_RawData.cav
* Machine Learning: 
   - Mass Shooting dataset: [USMassShootings07042022.csv](https://github.com/linxiuyun93/AIT580_Team6_Final-Project/blob/main/ML/USMassShootings07042022.csv)
   - Gun Law Score Card: [GunLawScoreCard.csv](https://github.com/linxiuyun93/AIT580_Team6_Final-Project/blob/main/ML/GunLawScoreCard.csv)
* NLP

#### Machine Learning 
#### Executable script: [AIT580_Team6_Project_ML.py](https://github.com/linxiuyun93/AIT580_Team6_Final-Project/blob/main/ML/AIT580_Team6_Project_ML.py)

#### Python libraries: `pandas`, `numpy`, `matplotlib`, `scipy`, `seaborn`, `sklearn`

#### Machine Learning Models
##### 1.  Linear Regresssion
  - gun_law_strength_score vs. fatalities: `R-Squared = 0.00568`
  - gun_law_strength_score vs. gun_death_rate_per_100k: `R-Squared = 0.5436`
##### 2.  KNN Classifier
  - Initial attempt: classify `‘fatalities’` with all numeric features and `accuracy rate = 0.1539`
  - 2nd attempt: classify `‘fatalities’` with dropping `'injured'` and `accuracy rate = 0.1154`
  - 3rd attempt: classify `‘fatalities’` with dropping `'total_victims'` and `accuracy rate = 0.2308`
  - 4th attempt: classify `‘fatalities’` with dropping `'age_of_shooter'` and `accuracy rate = 0.1153`
  - 5th attempt: classify `‘fatalities’` with dropping `'race'` and `accuracy rate = 0.1154`
  - 6th attempt: classify `‘fatalities’` with dropping `'gender'` and `accuracy rate =0.0770 `
  - 7th attempt: classify `‘fatalities’` with dropping `'year'` and `accuracy rate = 0.1539`
  - 8th attempt: classify `‘fatalities’` with dropping `'gun_law_strength_score'` and `accuracy rate = 0.2308`
  - 9th attempt: classify `‘fatalities’` with dropping `'gun_death_rate_per_100k'` and `accuracy rate = 0.2692`
  - Adjustment: classify `‘fatalities’` with `'age_of_shooter'`,`'race'`,`'gender'`,`'gun_law_strength_score'` and `accuracy rate = 0.1923`

##### 3. Random Forest Classification
  - Accuracy rate = 0.1795

#### Conclusion: 
 Among Linear regression, KNN and Random forest classification, KNN provided the most high accuracy level. The result of identify the impact of gun law on massive shooting were quite dismal. No model was able to predict mass shootings with any level of accuracy greater than random chance. The relative infrequency of mass shooting can make cush correlations harder to dect. 

## NLP Analysis

