#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import seaborn as sns
import geopandas as gpd
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from shapely.geometry import Point, Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier as _KNN_Classify_features
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Read two data files
d1= pd.read_csv('USMassShootings07042022.csv')
d2=pd.read_csv('GunLawScoreCard.csv')


# In[3]:


#join two data files
d1['state']=d1['location'].str.split(',',expand=True)[1]
d1['state']=d1['state'].str.strip()
df=pd.merge(d1,d2, on='state',how='left')
df=df.drop(['gun_law_strength(ranked)'],axis=1)
df=df.drop(['gun_death_rate(ranked)'],axis=1)


# In[4]:


# Convert gun_law_strength_gradd to scores
df['gun_law_strength_score']=df['gun_law_strength_grade']

for i in range(len(df['gun_law_strength_score'])):
  
    # replace A with 4
    if df['gun_law_strength_score'][i] == 'A':
        df['gun_law_strength_score'][i] = 4
    # replace A- with 3.667
    if df['gun_law_strength_score'][i] == 'A-':
        df['gun_law_strength_score'][i] = 3.667
    # replace B+ with 3.333
    if df['gun_law_strength_score'][i] == 'B+':
        df['gun_law_strength_score'][i] = 3.333
    # replace B with 3
    if df['gun_law_strength_score'][i] == 'B':
        df['gun_law_strength_score'][i] = 3
    # replace B- with 2.667
    if df['gun_law_strength_score'][i] == 'B-':
        df['gun_law_strength_score'][i] = 2.667
    # replace C+ with 2.333
    if df['gun_law_strength_score'][i] == 'C+':
        df['gun_law_strength_score'][i] = 2.333
    # replace C with 2
    if df['gun_law_strength_score'][i] == 'C':
        df['gun_law_strength_score'][i] = 2
    # replace C- with 1.6667
    if df['gun_law_strength_score'][i] == 'C-':
        df['gun_law_strength_score'][i] = 1.6667
    # replace D+ with 1.333
    if df['gun_law_strength_score'][i] == 'D+':
        df['gun_law_strength_score'][i] = 1.333
    # replace D with 1
    if df['gun_law_strength_score'][i] == 'D':
        df['gun_law_strength_score'][i] = 1
    # replace D- with 0.667
    if df['gun_law_strength_score'][i] == 'D-':
        df['gun_law_strength_score'][i] = 0.667
    # replace F with 0
    if df['gun_law_strength_score'][i] == 'F':
        df['gun_law_strength_score'][i] = 0
        
df['gun_law_strength_score'] = pd.to_numeric(df['gun_law_strength_score'])


# In[5]:


# check unique values in gender column
df.gender.value_counts()


# In[6]:


# Convert gender feature to numerical number
for i in range(len(df['gender'])):
  
    # replace Male with 1
    if df['gender'][i] == 'Male':
        df['gender'][i] = 1
    # replace M with 1
    if df['gender'][i] == 'M':
        df['gender'][i] = 1
    # replace Fale with 2
    if df['gender'][i] == 'Female':
        df['gender'][i] = 2
    # replace F with 1
    if df['gender'][i] == 'F':
        df['gender'][i] = 2
    # replace Male & Female
    if df['gender'][i] == 'Male & Female':
        df['gender'][i] = 3
        
df['gender'] = pd.to_numeric(df['gender'])


# In[7]:


# conver capitalized value and replace special character and unclear value to Other
df['race'] = df['race'].str.upper()
df.race=df.race.str.strip()
df.race.value_counts()


# In[8]:


#Convert race to numberical 
for i in range(len(df['race'])):
  
    # replace - with white
    if df['race'][i] == '-':
        df['race'][i] = 'OTHER'
    if df['race'][i] == 'UNCLEAR':
        df['race'][i] = 'OTHER'
    if df['race'][i] == 'WHITE':
        df['race'][i] = 1
    if df['race'][i] == 'BLACK':
        df['race'][i] = 2
    if df['race'][i] == 'LATINO':
        df['race'][i] = 3
    if df['race'][i] == 'ASIAN':
        df['race'][i] = 4
    if df['race'][i] == 'NATIVE AMERICAN':
        df['race'][i] = 5
    if df['race'][i] == 'OTHER':
        df['race'][i] = 6


# In[9]:


#save the merged source file 
df.to_csv("merged_source.csv", index = False)


# In[10]:


#Calculate and draw a correlation matrix
df = df[['fatalities', 'injured', 'total_victims', 'age_of_shooter','gender','gun_law_strength_score','gun_death_rate_per_100k']]
df.corr('pearson')# set the size of the figure
plt.rcParams["figure.figsize"] = (12,12)

# use matplotlib.pyplot.matshow() to represent an correlation matrix in a new figure window
plt.matshow(df.corr())

# set the ticks
plt.xticks(range(len(df.columns)), df.columns, rotation=90)
plt.yticks(range(len(df.columns)), df.columns)

# set the color bar
plt.colorbar()

# draw
plt.show()


# In[11]:


df = df[['fatalities', 'injured', 'total_victims', 'age_of_shooter','gender','gun_law_strength_score','gun_death_rate_per_100k']]
df.corr('pearson')


# In[12]:


#Draw a scatter matrix
from pandas.plotting import scatter_matrix
axes = scatter_matrix(df, alpha=0.2, figsize = (12, 12))


# In[13]:


# data visualization 
import warnings
import datetime
from datetime import timedelta
import time
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")
d1= pd.read_csv('merged_source.csv')
d1=d1[['location','year','fatalities', 'injured', 'total_victims','incident_location','state','prior_signs_mental_health_issues','weapons_obtained_legally','where_obtained','weapon_type','age_of_shooter','gender','region','race','gun_law_strength_score','gun_death_rate_per_100k']]
# create list of columns to iterate over for buttons
cols = d1.columns.values.tolist()
# make list of default plotly colors in hex
plotly_colors=[
                '#1f77b4',  # muted blue
                '#ff7f0e',  # safety orange
                '#2ca02c',  # cooked asparagus green
                '#d62728',  # brick red
                '#9467bd',  # muted purple
                '#8c564b',  # chestnut brown
                '#e377c2',  # raspberry yogurt pink
                '#7f7f7f',  # middle gray
                '#bcbd22',  # curry yellow-green
                '#17becf'   # blue-teal
              ]
# create dictionary to associate colors with unique categories
color_dict = dict(zip(d1['gender'].unique(),plotly_colors))
# map new column with hex colors to pass to go.Scatter()
d1['hex']= d1['gender'].map(color_dict)
#initialize scatter plot
fig = go.Figure(
    go.Scatter(
        x=d1['gun_law_strength_score'],
        y=d1['fatalities'],
        text=d1['injured'],
        marker=dict(color=d1['hex']),
        mode="markers"
    )
) 
# initialize dropdown menus
fig.update_layout(
    updatemenus=[
        {
            "buttons": [
                {
                    "label": f"x - {x}",
                    "method": "update",
                    "args": [
                        {"x": [d1[x]]},
                        {"xaxis": {"title": x}},
                    ],
                }
                for x in cols
            ]
        },
        {
            "buttons": [
                {
                    "label": f"y - {x}",
                    "method": "update",
                    "args": [
                        {"y": [d1[x]]},
                        {"yaxis": {"title": x}}
                    ],
                }
                for x in cols
            ],
            "y": 0.9,
        },
    ],
    margin={"l": 0, "r": 0, "t": 25, "b": 0},
    height=250

)
fig.show()


# In[14]:


#four features which have high Pearson's correlation coefficients with gun_law_strength_score
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
fig.suptitle('Comparing Paris for Several Features', size=15)

# plot 1st one 
df1 = df[['gun_law_strength_score','fatalities']]
ax1.scatter(df1.gun_law_strength_score, df1.fatalities, c='orange', marker='o')
ax1.set_xlabel('gun_law_strength_score', fontsize=10)
ax1.set_ylabel('fatalities', fontsize=10)
ax1.set_ylim([0, 30])
ax1.title.set_text('fatalities vs gun_law_strength_score')

# plot 2nd one
df2 = df[['gun_law_strength_score','injured']]
ax2.scatter(df2.gun_law_strength_score, df2.injured, c='Yellow', marker='o')
ax2.set_xlabel('gun_law_strength_score', fontsize=10)
ax2.set_ylabel('injured', fontsize=10)
ax2.set_ylim([0, 30])
ax2.title.set_text('injured vs gun_law_strength_score')

# plot 3rd one
df3 = df[['gun_death_rate_per_100k','gun_law_strength_score']]
ax3.scatter(df3.gun_law_strength_score, df3.gun_death_rate_per_100k, c='lightblue', marker='o')
ax3.set_xlabel('gun_law_strength_score', fontsize=10)
ax3.set_ylabel('gun_death_rate_per_100k', fontsize=10)
ax3.set_ylim([0, 30])
ax3.title.set_text('gun_death_rate(per_100k) vs gun_law_strength_score')

# plot 4th one
df4 = df[['total_victims','gun_law_strength_score']]
ax4.scatter(df4.gun_law_strength_score, df4.total_victims, c='green', marker='o')
ax4.set_xlabel('gun_law_strength_score', fontsize=10)
ax4.set_ylabel('total_victims', fontsize=10)
ax4.set_ylim([0, 30])
ax4.title.set_text('total_victims vs gun_law_strength_score')

plt.show()


# In[15]:


#Linear Regression Model 
#y= mx + b >> in our case Weight = MPG * m + b
# Creatubg X and y
X=df[['gun_law_strength_score']].values
y=df['gun_death_rate_per_100k']


# In[16]:


#Split the dataset into a training set (80%) and a test set (20%)
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size= 0.8, test_size=0.2, random_state=100)
print('Total rows: ', len(df))
print('Training rows: ', len(X_train))
print('Testing rows: ', len(X_test))


# In[17]:


#Build a LR model to find details of the relationship between gun_law_strength_score and gun_death_rate_per_100k.
X = df[['gun_law_strength_score']].values
y = df['gun_death_rate_per_100k']

model = LinearRegression(fit_intercept=False)
clf = model.fit(X, y)
print ('Coefficient: ', clf.coef_)

predictions = model.predict(X)
#for index in range(len(predictions)):
for index in range(10):
  print('Actual: ', y[index], 'Predicted: ', predictions[index], 'gun_law_strength_score: ', X[index,0])


# In[18]:


# Create linear regression object
model = LinearRegression()
# Train the model using the training sets
model.fit(X_train, y_train)
r_sq = model.score(X, y)
print(f"coefficient of determination: {r_sq}")
print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}")


# In[19]:


#Use the test set to make prediction and print out Root MeanSquared Error (RMSE).
y_pred = model.predict(X_test)
df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
print(df_preds)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))


# In[20]:


#Plot the predicted straight line with the test data
prediction_space = np.linspace(min(X_test), max(X_test)).reshape(-1,1)

# Compute predictions over the prediction space: y_pred
y_pred = model.predict(prediction_space)

# Plot regression line
plt.plot(prediction_space, y_pred, color='blue', linewidth=2)
plt.scatter(df.gun_law_strength_score, df.gun_death_rate_per_100k, c='green', marker='o')
plt.show()


# In[21]:


#Build a LR model to find details of the relationship between gun_law_strength_score and fatalities
df=pd.read_csv('merged_source.csv')
# y= mx + b >> in our case Weight = MPG * m + b
# Creatubg X and y
X=df[['gun_law_strength_score']].values
y=df['fatalities']


# In[22]:


#Split the dataset into a training set (80%) and a test set (20%)
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size= 0.8, test_size=0.2, random_state=100)
print('Total rows: ', len(df))
print('Training rows: ', len(X_train))
print('Testing rows: ', len(X_test))


# In[23]:


#Build a LR model to find details (i.e., intercept and slope) of the relationship between gun_law_strength_score and fatalities
X = df[['gun_law_strength_score']].values
y = df['total_victims']

model = LinearRegression(fit_intercept=False)
clf = model.fit(X, y)
print ('Coefficient: ', clf.coef_)

predictions = model.predict(X)
#for index in range(len(predictions)):
for index in range(10):
  print('Actual: ', y[index], 'Predicted: ', predictions[index], 'gun_law_strength_score: ', X[index,0])


# In[24]:


# Create linear regression object
model = LinearRegression()
# Train the model using the training sets
model.fit(X_train, y_train)
r_sq = model.score(X, y)
print(f"coefficient of determination: {r_sq}")
print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}")


# In[25]:


#Use the test set to make prediction and print out Root MeanSquared Error (RMSE).
y_pred = model.predict(X_test)
df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
print(df_preds)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))


# In[26]:


#Plot the predicted straight line with the test data
prediction_space = np.linspace(min(X_test), max(X_test)).reshape(-1,1)

# Compute predictions over the prediction space: y_pred
y_pred = model.predict(prediction_space)

# Plot regression line
plt.plot(prediction_space, y_pred, color='blue', linewidth=2)
plt.scatter(df.gun_law_strength_score, df.total_victims, c='orange', marker='o')
plt.xlim(0, 4)
plt.ylim(0, 30)
plt.show()


# In[56]:


# K-mean clustering 
# standardizing the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df=pd.read_csv('merged_source.csv')
X=df[['gun_law_strength_score','fatalities']].values
X= scaler.fit_transform(X)

# statistics of scaled data
pd.DataFrame(X).describe()


# In[57]:


# Using the elbow method to find the optimal number of clusters
kmeans = KMeans(n_clusters=3, init='k-means++')

# fitting the k means algorithm on scaled data
kmeans.fit(X)


# In[58]:


kmeans.inertia_


# In[59]:


SSE = []
for cluster in range(1,20):
    kmeans = KMeans(n_clusters = cluster, init='k-means++')
    kmeans.fit(X)
    SSE.append(kmeans.inertia_)

# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')


# In[60]:


# k means using 4 clusters and k-means++ initialization
kmeans = KMeans( n_clusters = 4, init='k-means++')
kmeans.fit(X)
pred = kmeans.predict(X)
frame = pd.DataFrame(X)
frame['cluster'] = pred
frame['cluster'].value_counts()


# In[61]:


y_kmeans = kmeans.fit_predict(X)
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 60, c = 'red', label = 'Cluster1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 60, c = 'blue', label = 'Cluster2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 60, c = 'green', label = 'Cluster3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 60, c = 'violet', label = 'Cluster4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 60, c = 'orange', label = 'Cluster5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'black', label = 'Centroids')
plt.xlabel('gun_law_strength_score') 
plt.ylabel('fatalities')
plt.legend() 
plt.show()


# In[62]:


# hierarchical clustering
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
# create dendrogram
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
# create clusters
hc = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'ward')
# save clusters for chart
y_hc = hc.fit_predict(X)


# In[63]:


#KNN Classifier to classify ‘fatalities’ with all numeric features. Print the model accuracy score and confusion matrix.
#must encode Origin labels for training
df=pd.read_csv('merged_source.csv')
le = preprocessing.LabelEncoder()
le.fit(df['fatalities'])
print(); print(list(le.classes_))
print(); print(le.transform(df['fatalities']))


# In[68]:


X = df[['injured','total_victims','age_of_shooter','race','gender','year','gun_law_strength_score','gun_death_rate_per_100k']].values
y = df['fatalities'].values
print(X.shape, y.shape)


# In[69]:


# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


# In[70]:


knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train,y_train)


# In[71]:


y_pred = knn.predict(X_test)
print("Predictions: {}".format(y_pred)) 


# In[72]:


print("Accuracy: ", knn.score(X_test, y_test))
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))
print("Classification Report: ")
print(classification_report(y_test, y_pred))


# In[74]:


print(df.corr())
plt.matshow(df.corr())


# In[92]:


# ajust the KNN model to have higher accuracy value
X = df[['age_of_shooter','race','gender','gun_law_strength_score']].values
y = df['fatalities'].values

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train,y_train)


# In[93]:


y_pred = knn.predict(X_test)
print("Predictions: {}".format(y_pred)) 
print("Accuracy: ", knn.score(X_test, y_test))
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))
print("Classification Report: ")
print(classification_report(y_test, y_pred))


# In[95]:


# randmon forest classification
import pandas as pd
import numpy as np
data= pd.read_csv('merged_source.csv')
X=data[['gun_law_strength_score','injured','total_victims','age_of_shooter','race','gender']].values
y=data[['fatalities']].values
X


# In[96]:


from sklearn import preprocessing
from sklearn import utils

#convert y values to categorical values
lab = preprocessing.LabelEncoder()
y= lab.fit_transform(y)


#view transformed values
print(y)


# In[98]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


# In[99]:


from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(criterion='entropy')   
rf_clf.fit(X_train,y_train)


# In[100]:


RandomForestClassifier(criterion='entropy')
y_predict = rf_clf.predict(X_test)


# In[101]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
accuracy_score(y_test,y_predict)





