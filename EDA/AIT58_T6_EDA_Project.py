import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns


# import geopandas as gpd
# from shapely.geometry import Point, Polygon
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# %matplotlib inline

# ##################################################################################
# 1. In which year we had the most shooting from 1982 - 2022? - Smooth line chart
# In every five years, we can see there is a spike in mass shooting among which the highest was between 2015-2022.
# The fatalities increased a lot in these 5 years. We had these spikes in the 5 years interval from 1982 - 2022
# ###################################################################################
# 2. Fatalities as per location?
# We can see that the fatalities as per location has changed dramatically since 1982.
# ###################################################################################
# 3. Fatalities and injuries per year
# Shotting and fatalities were at its highest in 2017. 
# ###################################################################################
# 4. Which age group was most involved in the shooting? - Bar plot
# Early and mid 20s along with mid 60s have the most number of shooters. 
# ###################################################################################
# 5. Fatalities as per location
# Workplace and school have the most fatality rates among all the data we have. 
# ###################################################################################
# 6. Comparing total victims to fatalities and injured as per states - Subburst chart
# We can see that Florida, California has more number of fatalities as compared to Nevada and Texas whereas total 
# victims is more in Nevada and Texas as compared to these both. 
# ###################################################################################
# 7. Which state has the more number of fatalities?
# California and texas
# ###################################################################################
# 8. Check if there is any correlation between gun law, year, fatalities, injured and total victims
# No there is none. 
# ###################################################################################


# Data loading and cleaning
data = pd.read_csv(r'USMassShootings_RawData.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data = data.replace({'gender':{'Male':'M','Female':'F', 'Male & Female':'M/F'}}, regex=True)

data = data.replace({'incident_location':'workplace'},{'incident_location':'Workplace'})
data = data.replace({'incident_location':'\r\nWorkplace'},{'incident_location':'Workplace'})
data = data.replace({'incident_location':'Other\r\n'},{'incident_location':'Other'})

shooting_as_per_year = data[['fatalities','injured','total_victims','incident_location','year']]


def shooting_yearly():

	sum_shooting = shooting_as_per_year.groupby('year').sum()
	# sum_shooting.sort_values('total_victims', ascending=False).head(5)

	plt.rcParams["figure.figsize"] = [10, 5]
	plt.rcParams["figure.autolayout"] = True


	# Set x and Y
	X = sum_shooting.index
	x = sum_shooting['fatalities']
	y = sum_shooting['injured']
	z = sum_shooting['total_victims']

	plt.plot(X, x, label='Fatalities')
	plt.plot(X, y, label='Injured')
	plt.plot(X, z, label='Total victims')

	plt.legend(loc='upper left')
	plt.xlabel('Year')
	plt.ylabel('Fatalities Injured and total victims')

	plt.title('Year vs Fatalities Injured and total victims')

	plt.grid()


def sum_shooting_yealy():

	sum_shoot = shooting_as_per_year.groupby(['incident_location','year'], as_index=False).sum().sort_values('total_victims',ascending=False).head(20)
	# sum_shoot

	plt.rcParams["figure.figsize"] = [20, 10]
	plt.rcParams["figure.autolayout"] = True

	# # Set x and Y
	X = sum_shoot.year
	y = sum_shoot['fatalities']


	sns.barplot(data=sum_shoot, x=X, y=y, hue=sum_shoot.incident_location)

	plt.grid()

	plt.title('Fatalitles VS Year along with their location')

	plt.show()


def fatalities_and_injuries_yearly():

	fatalities = data.groupby("year")["fatalities"].sum()
	fatalities

	x = fatalities.index
	y = fatalities.values
	plt.figure(figsize=(20,6))
	sns.barplot(x=x, y=y)
	plt.xlabel("Fatalities per year", fontsize=20)
	plt.ylabel("Count", fontsize=20)

	injuries = data.groupby("year")["injured"].sum()
	injuries

	plt.figure(figsize=(20,6))
	sns.barplot(x=injuries.index, y=injuries.values)
	plt.xlabel("Injuries per year", fontsize=20)
	plt.ylabel("Count", fontsize=20)


def shooting_age():

	shooting_as_per_age = data[['fatalities','injured','total_victims','incident_location','age_of_shooter','gender']]
	# shooting_as_per_age

	shooter_age = shooting_as_per_age.groupby('age_of_shooter',as_index=False).sum().sort_values('fatalities',ascending=False)
	shooter_age.head()

	plt.rcParams["figure.figsize"] = [20, 20]
	plt.rcParams["figure.autolayout"] = True

	# Set x and Y
	X = shooter_age.age_of_shooter
	y = shooter_age['fatalities']

	sns.barplot(data=shooter_age, x=X, y=y)
	plt.grid()
	plt.title('Fatalities VS Age')

# Random correlation trial

# df = shooter_age.drop(shooter_age.index[shooter_age['age_of_shooter'] == '-'], inplace=True)
# shooter_age = shooter_age.astype({'age_of_shooter':'int'})
# # shooter_age['age_of_shooter'].unique()

# shooter_age.corr()

# No valuable output here


def shooting_location():

	shooter_location = data[['age_of_shooter','incident_location','fatalities']].groupby(['incident_location','age_of_shooter'],as_index=False).sum().sort_values(['incident_location','fatalities']).groupby('incident_location')['fatalities'].max()

	shooter_location.columns = ['location','fatalities']

	df = pd.DataFrame(shooter_location, columns=['fatalities'])

	plt.figure(figsize=(10,10))
	plt.pie(df.fatalities, textprops={'fontsize': 20}, autopct='%1.1f%%', labels = df.index, explode = [0.0, 0.0, 0.2, 0.0, 0.0, 0.1], colors = ['#584DA1','#4D96A1','#50F7D6','#50A458','#A4A450','#189BF2'])
	plt.legend()
	plt.title('Fatalities as per location',pad='50.0',fontsize = 40)
	plt.show()


def sunburst_chart():
	res = data.groupby(['State'],as_index=False).sum().sort_values('total_victims',ascending=False).head(5)
	res

	fig = px.sunburst(res, path=['State', 
	                            'fatalities',
	                            'injured',
	                            'total_victims'], 
	                  values='fatalities',color= 'fatalities',title="Fatalities, Injuries and total victims for top 5 states")
	fig.show()

def map_charts():

	gun_law_df = data[['gun_law_strength_score','fatalities','injured','total_victims','State','latitude','longitude','State_abb']].groupby(['State','gun_law_strength_score'],as_index=False).sum()
	gun_law_df_corr = gun_law_df[['gun_law_strength_score','fatalities','injured','total_victims']].corr(method ='pearson')

	state_wise_shooting = data[['year','fatalities','injured','total_victims','State','latitude','longitude','State_abb']].groupby('State_abb',as_index=False).sum()

	fig1 = px.choropleth(state_wise_shooting,
	                    locations='State_abb', 
	                    locationmode="USA-states", 
	                    scope="usa" , color='fatalities',
	                    color_continuous_scale="Viridis_r"                   
	                    )
	fig2 = px.choropleth(state_wise_shooting,
	                    locations='State_abb', 
	                    locationmode="USA-states", 
	                    scope="usa" , color='injured',
	                    color_continuous_scale="Viridis_r"                   
	                    )
	fig3 = px.choropleth(state_wise_shooting,
	                    locations='State_abb', 
	                    locationmode="USA-states", 
	                    scope="usa" , color='total_victims',
	                    color_continuous_scale="Viridis_r"                   
	                    )

	fig1.update_layout(
	      title_text = 'Fatalities as per US state',
	      title_font_family="Times New Roman",
	      title_font_size = 22,
	      title_font_color="black", 
	      title_x=0.45, 
	         )
	fig2.update_layout(
	      title_text = 'Injured as per US state',
	      title_font_family="Times New Roman",
	      title_font_size = 22,
	      title_font_color="black", 
	      title_x=0.45, 
	         )
	fig3.update_layout(
	      title_text = 'Total victims as per US state',
	      title_font_family="Times New Roman",
	      title_font_size = 22,
	      title_font_color="black", 
	      title_x=0.45, 
	         )
	fig1.show()
	fig2.show()
	fig3.show()


	year_wise_state = data[['year','fatalities','injured','total_victims','State_abb']].groupby(['State_abb','year'],as_index=False).sum().sort_values('year')

	fig1 = px.choropleth(year_wise_state,
	                    locations='State_abb', 
	                    locationmode="USA-states", 
	                    scope="usa" , color='fatalities',
	                    color_continuous_scale="Viridis_r",
	                    animation_frame='year'
	                    )
	fig2 = px.choropleth(year_wise_state,
	                    locations='State_abb', 
	                    locationmode="USA-states", 
	                    scope="usa" , color='injured',
	                    color_continuous_scale="Viridis_r",
	                    animation_frame='year'
	                    )
	fig3 = px.choropleth(year_wise_state,
	                    locations='State_abb', 
	                    locationmode="USA-states", 
	                    scope="usa" , color='total_victims',
	                    color_continuous_scale="Viridis_r",
	                    animation_frame='year'
	                    )

	fig1.update_layout(
	      title_text = 'Fatalities as per state from 1982 - 2022',
	      title_font_family="Times New Roman",
	      title_font_size = 22,
	      title_font_color="black", 
	      title_x=0.45, 
	         )
	fig2.update_layout(
	      title_text = 'Injured as per state from 1982 - 2022',
	      title_font_family="Times New Roman",
	      title_font_size = 22,
	      title_font_color="black", 
	      title_x=0.45, 
	         )
	fig3.update_layout(
	      title_text = 'Total victims as per state from 1982 - 2022',
	      title_font_family="Times New Roman",
	      title_font_size = 22,
	      title_font_color="black", 
	      title_x=0.45, 
	         )
	fig1.show()
	fig2.show()
	fig3.show()


	year_wise_state_law = data[['year','fatalities','injured','total_victims','gun_law_strength_score','State_abb']].groupby(['State_abb','year'],as_index=False).sum().sort_values('year')

	fig = px.choropleth(year_wise_state_law,
	                    locations='State_abb', 
	                    locationmode="USA-states", 
	                    scope="usa" , color='gun_law_strength_score',
	                    color_continuous_scale="Viridis_r",
	                    animation_frame='year'
	                    )

	fig.update_layout(
	      title_text = 'gun_law_strength_score as per state from 1982 - 2022',
	      title_font_family="Times New Roman",
	      title_font_size = 22,
	      title_font_color="black", 
	      title_x=0.45, 
	         )

	fig.show()

def correlation_checks():	

	sns.set(rc = {'figure.figsize':(10,5)})

	sns.heatmap(gun_law_df_corr,cmap="YlGnBu", annot=True)


	gun_law__yearly_df = data[['year','gun_law_strength_score','fatalities','injured','total_victims','State','latitude','longitude','State_abb']].groupby(['year'],as_index=False).sum()
	gun_law__yearly_df_corr = gun_law__yearly_df[['year','fatalities','injured','total_victims']].corr(method ='pearson')
	gun_law__yearly_df_corr

	sns.set(rc = {'figure.figsize':(10,5)})

	sns.heatmap(gun_law__yearly_df_corr,cmap="YlGnBu", annot=True)


	fig, axes = plt.subplots(3, 2, figsize=(18, 10))

	sns.regplot(ax=axes[0][0],y=gun_law__yearly_df[['fatalities']],x=gun_law__yearly_df[['year']])
	sns.regplot(ax=axes[1][0],y=gun_law__yearly_df[['injured']],x=gun_law__yearly_df[['year']])
	sns.regplot(ax=axes[2][0],y=gun_law__yearly_df[['total_victims']],x=gun_law__yearly_df[['year']])
	sns.regplot(ax=axes[0][1],x=age_df[['year']],y=age_df[['age_of_shooter']])


	age_df = data[['year','age_of_shooter','fatalities','injured','total_victims','State','latitude','longitude','State_abb']].groupby(['age_of_shooter','year'],as_index=False).sum()
	age_df_corr = age_df[['year','age_of_shooter','fatalities','injured','total_victims']].corr(method ='pearson')
	age_df_corr

	sns.heatmap(age_df_corr,cmap="YlGnBu", annot=True)



def main():
    
    shooting_yearly()
    sum_shooting_yealy()
    fatalities_and_injuries_yearly()
    shooting_age()
    shooting_location()
    sunburst_chart()
    map_charts()
    correlation_checks()

if __name__ == "__main__":
    main()