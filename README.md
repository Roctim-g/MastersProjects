# AIT 580 Final Project:<br/>Understanding US Mass Shooting Data Last 50 Years
  Team 6: Roctim Gogoi, Nina Lin, Tony Sanchez
## Sytem Set Up

## Dataset Links
* EDA:
   - Mass shooting dataset: [USMassShootings_RawData.csv](https://github.com/linxiuyun93/AIT580_Team6_Final-Project/blob/main/EDA/USMassShootings_RawData.csv) 
* Machine Learning: 
   - Mass Shooting dataset: [USMassShootings07042022.csv](https://github.com/linxiuyun93/AIT580_Team6_Final-Project/blob/main/USMassShootings07042022.csv)
   - Gun Law Score Card: [GunLawScoreCard.csv](https://github.com/linxiuyun93/AIT580_Team6_Final-Project/blob/main/GunLawScoreCard.csv)
* NLP
   - Updated Mass Shooting dataset: [USMassShootings05242022_update0704.csv] (https://github.com/linxiuyun93/AIT580_Team6_Final-Project/blob/main/NLP/USMassShootings05242022_update0704.csv) 

## Exploratory Data Analysis
Executable script: [AIT580_T6_EDA_Project.py](https://github.com/linxiuyun93/AIT580_Team6_Final-Project/blob/main/NLP/USMassShootings_RawData.csv)  

Python libraries: pandas, numpy, matplotlib.pyplot, plotly.express, seaborn

|Visualization result|
|---------------------|
|Some parts of the states have a high mass shooting rate as commpared to other states. From columns and geo charts, we saw that california is having the highest fatality and nevada has the highest injury rate as comapred to other states.|
|There is a correlation between year and the injury and fatalities. Year to fatality is 0.5, where year to injury 0.32 and year to total_victims 0.39.|
|There is a negative correlation between year and the age of the shooter which means as the year proceed from 1982, the age of the shooter who is involved in the shooting has come closer to younsters.|


## Machine Learning
### Executable script: [AIT580_Team6_Project_ML.py](https://github.com/linxiuyun93/AIT580_Team6_Final-Project/blob/main/ML/AIT580_Team6_Project_ML.py) 

### Python libraries: `pandas`, `numpy`, `matplotlib`, `scipy`, `seaborn`, `sklearn`

### Machine Learning Models
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

### Conclusion: 
Among Linear regression, KNN and Random forest classification, KNN provided the most high accuracy level. The result of identify the impact of gun law on massive shooting were quite dismal. No model was able to predict mass shootings with any level of accuracy greater than random chance. The relative infrequency of mass shooting can make cush correlations harder to dect. 



## NLP Analysis

To run the NLP Code, You'll need at least python 3.8, 3.10 is preferred to run NER draw NER Chunk:

### Executable script: [usMassShootingsNLP_Run.py](https://github.com/linxiuyun93/AIT580_Team6_Final-Project/blob/main/NLP/usMassShootingsNLP_Run.py) 

### Python libraries: `pandas`, `numpy`, `matplotlib`, `scipy`, `seaborn`, `sklearn`, `nltk`, `collections`, `nltk.corpus`, `gensim`, `re`, `pprint`, `wordcloud`, `PIL`, `itertools`, `spacy` 

```javascript
Imported libraries specific modules include:
from gensim.corpora.dictionary import Dictionary
from collections import defaultdict

import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from collections import Counter
from nltk.corpus import stopwords
from gensim.models.tfidfmodel import TfidfModel
from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

import numpy as np
import matplotlib.pyplot as plt

import re, pprint
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import pandas as pd
from PIL import Image
import itertools
import spacy
from sklearn.feature_extraction.text import CountVectorizer
```



To run the code in a terminal, Run the command:
/usr/bin/python3 usMassShootingsNLP_Run.py USMassShootings05242022_update0704.csv

