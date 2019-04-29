
# coding: utf-8

# <h1 align="center"> 
# DATS 6501 —Capstone Project (Part I)
# </h1> 
# 
# <h1 align="center"> 
# 120 Years Olympic Games Analysis
# </h1> 
# 
# <h4 align="center"> 
# Author: Xiaochi Ge ([gexiaochi@gwu.edu](mailto:gexiaochi@gwu.edu))
# </h4>

# ## 1. Import Packages & Read Data

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


athletes = pd.read_csv('athlete_events.csv')
regions = pd.read_csv('noc_regions.csv')


# In[3]:


athletes.head()


# In[4]:


regions.head()


# In[5]:


olympic_data = pd.merge(athletes, regions, on='NOC', how='left')
olympic_data.head()


# In[6]:


olympic_data.info()


# In[7]:


#missing data
olympic_data.isnull().sum()


# ## 2. Distribution of Age, Height and Weight

# In[8]:


#Max and Min Age
MaxAge = olympic_data.Age.max()
MaxAge = int(round(MaxAge))

MinAge = olympic_data.Age.min()
MinAge = int(round(MinAge))

AvgAge = olympic_data.Age.mean()
AvgAge = int(round(AvgAge))

MedAge = olympic_data.Age.median()
MedAge = int(round(MedAge))

print('The maximum age during 120 years olympic history is:', MaxAge, "years old")
print('The minimum age during 120 years olympic history is:', MinAge, "years old")
print('The average age during 120 years olympic history is:', AvgAge, "years old")
print('The median age during 120 years olympic history is:', MedAge, "years old")


# In[9]:


#Max and Min Height
MaxHeight = olympic_data.Height.max()
MaxHeight = int(round(MaxHeight))

MinHeight = olympic_data.Height.min()
MinHeight = int(round(MinHeight))

AvgHeight = olympic_data.Height.mean()
AvgHeight = int(round(AvgHeight))

MedHeight = olympic_data.Height.median()
MedHeight = int(round(MedHeight))

print('The maximum height during 120 years olympic history is:', MaxHeight, "cm")
print('The minimum height during 120 years olympic history is:', MinHeight, "cm")
print('The average height during 120 years olympic history is:', AvgHeight, "cm")
print('The median height during 120 years olympic history is:', MedHeight, "cm")


# In[10]:


#Max and Min Weight
MaxWeight = olympic_data.Weight.max()
MaxWeight = int(round(MaxWeight))

MinWeight = olympic_data.Weight.min()
MinWeight = int(round(MinWeight))

AvgWeight = olympic_data.Weight.mean()
AvgWeight = int(round(AvgWeight))

MedWeight = olympic_data.Weight.median()
MedWeight = int(round(MedWeight))

print('The maximum weight during 120 years olympic history is:', MaxWeight, "kg")
print('The minimum weight during 120 years olympic history is:', MinWeight, "kg")
print('The average weight during 120 years olympic history is:', AvgWeight, "kg")
print('The median weight during 120 years olympic history is:', MedWeight, "kg")


# In[11]:


#replace missing value with median value
olympic_data['Age'].fillna((olympic_data['Age'].median()), inplace=True)
olympic_data['Height'].fillna((olympic_data['Height'].median()), inplace=True)
olympic_data['Weight'].fillna((olympic_data['Weight'].median()), inplace=True)


# In[12]:


#Distribution of Age, Height, and Weight
f,ax=plt.subplots(figsize=(8,10))

plt.subplot(311)
sns.distplot(olympic_data['Age'],color='Red',kde=True)

plt.subplot(312)
sns.distplot(olympic_data['Height'],color='Blue',kde=True)

plt.subplot(313)
sns.distplot(olympic_data['Weight'],color='Green',kde=True)

plt.subplots_adjust(wspace = 0.2, hspace = 0.6,top = 0.9)
plt.show()


# ## 3. Fill-in Missing Data

# In[13]:


print(olympic_data['Medal'].value_counts(dropna=False))


# In[14]:


olympic_data['Medal'].fillna(('No_Medal'), inplace=True)
print(olympic_data['Medal'].value_counts(dropna=False))


# In[15]:


olympic_data.head()


# ## 4. Correlation Between Height, Weight and Age

# In[16]:


#weight and height
#sns.jointplot(x="Weight", y="Height", data=olympic_data)
sns.pairplot(olympic_data[['Height', 'Weight', 'Age']], kind="reg")
plt.show()


# ## 5. Check Host Years, Types of Sports, and Teams

# In[17]:


#check Years in olympic game history
print(np.sort(olympic_data.Year.unique()))


# In[18]:


#check type of sports in olympic game history
print(np.sort(olympic_data.Sport.unique()))


# In[19]:


print(olympic_data.region.unique())


# In[20]:


print(olympic_data.NOC.unique())


# In[21]:


print(olympic_data.Team.unique())


# In[22]:


olympic_data.loc[olympic_data['region'].isnull(),['NOC', 'Team']].drop_duplicates()


# ## 6. Comparison Between Summer and Winter Games

# In[23]:


#number of countries participated in summer
olympic_data[olympic_data['Season']=='Summer'].groupby('Year')['region'].nunique()


# In[24]:


#number of countries participated in winter
olympic_data[olympic_data['Season']=='Winter'].groupby('Year')['region'].nunique()


# In[100]:


sports=olympic_data.groupby(['Year','Season']).Sport.nunique().to_frame().reset_index()

plt.figure(figsize=(10,10))
ax=sns.pointplot(x=sports['Year'],y=sports['Sport'],hue=sports['Season'],dodge=True,palette="Set1")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_xlabel('Year',fontsize=10)
ax.set_ylabel('Count',fontsize=10)
ax.set_title('Types of Sports in Olympics',fontsize=16)
plt.show()


# In[26]:


#Summer game
summer = olympic_data[(olympic_data.Season== 'Summer')]
#Winter game
winter = olympic_data[(olympic_data.Season== 'Winter')]


# In[103]:


#popular Sports in Summer
plt.figure(figsize=(15,20))

plt.subplot(311)
summer_sport = summer['Sport'].value_counts().head(20)
summer_sport.plot(kind='bar', stacked=False, color = 'orangered')
plt.ylabel('Number of Sports')
plt.title('Top 20 Sports in Summer Olympic')

#popular Sports in Winter
plt.subplot(312)
winter_sport = winter['Sport'].value_counts().head(20)
winter_sport.plot(kind='bar', stacked=False, color = 'cornflowerblue')
plt.ylabel('Number of Sports')
plt.title('Top 20 Types of Sports in Winter Olympic')

plt.subplots_adjust(wspace = 0.2, hspace = 1,top = 1)
plt.show()


# In[104]:


#Popular Events in Summer
plt.figure(figsize=(15,20))

plt.subplot(311)
summer_event = summer['Event'].value_counts().head(10)
summer_event.plot(kind='bar', stacked=False, color = 'orangered')
plt.ylabel('Number of Events')
plt.title('Top 10 Events in Summer Olympic')

#Popular Events in Winter
plt.subplot(312)
winter_event = winter['Event'].value_counts().head(10)
winter_event.plot(kind='bar', stacked=False, color = 'cornflowerblue')
plt.ylabel('Number of Events')
plt.title('Top 10 Types of Events in Winter Olympic')

plt.subplots_adjust(wspace = 0.2, hspace = 1,top = 1)
plt.show()


# ## 7. Comparison Between Female and Male Athelets

# In[111]:


#Compare gender variation in Summer and Winter games
sum_gender = summer.groupby(["Year","Sex"])["ID"].nunique().reset_index()
win_gender = winter.groupby(["Year","Sex"])["ID"].nunique().reset_index()

fig = plt.figure(figsize=(14,14))
plt.subplot(211)
ax = sns.pointplot(x = sum_gender["Year"] , y = sum_gender["ID"], hue = sum_gender["Sex"], color='Orangered')

plt.ylabel("Number of Athletes")
plt.title("Variation of Gender in Summer Olympics")
plt.legend(loc = "best")

plt.subplot(212)
ax1 = sns.pointplot(x = win_gender["Year"] , y = win_gender["ID"],hue = win_gender["Sex"], color='midnightblue')

plt.ylabel("Number of Athletes")
plt.title("Variation of Gender Winter Olympics")
plt.legend(loc = "best")
plt.show()


# ## 8. Performances of Countries/Teams

# In[112]:


# Number of Teams participated in Summer
plt.figure(figsize=(14,14))
plt.subplot(211)
ax = summer.groupby("Year")["Team"].nunique().plot(kind = "line", color = 'orangered')
plt.ylabel("count")
plt.title("Number of Teams participated in Summer Olympic")

# Number of Teams participated in Winter
plt.subplot(212)
ax1 = winter.groupby("Year")["Team"].nunique().plot(kind = "line", color = 'cornflowerblue')
plt.title("Number of Teams participated in Winter Olympic")
plt.show()


# In[37]:


#number of medals by each team in Summer
medal_summer=summer.groupby(['NOC','Medal'])['Medal'].count()
medal_summer=medal_summer.unstack(level=-1,fill_value=0).reset_index()
#medal_summer.head()

#number of medals by each team in Winter
medal_winter=winter.groupby(['NOC','Medal'])['Medal'].count()
medal_winter=medal_winter.unstack(level=-1,fill_value=0).reset_index()
medal_winter.head()


# In[38]:


#total medals by each team in Summer
medal_summer['Total_Medals']=medal_summer['Bronze']+medal_summer['Gold']+medal_summer['Silver']
#number of Sports participated by each team in Summer
Sports_in_total_summer=summer.groupby('NOC')['Sport'].nunique().to_frame().reset_index()
#Sports_in_total_summer.head()

#total medals by each team in Winter
medal_winter['Total_Medals']=medal_winter['Bronze']+medal_winter['Gold']+medal_winter['Silver']
#number of Sports participated by each team in Winter
Sports_in_total_winter=winter.groupby('NOC')['Sport'].nunique().to_frame().reset_index()
#Sports_in_total_winter.head()


# In[39]:


#total medals by each team in Summer
medal_summer=pd.merge(medal_summer,Sports_in_total_summer,how='left',on='NOC')
medal_summer.sort_values('Total_Medals',ascending=False,inplace=True)
medal_summer=medal_summer[['NOC','Gold','Silver','Bronze','Total_Medals']]
#medal_summer.head()

#total medals by each team in Winter
medal_winter=pd.merge(medal_winter,Sports_in_total_winter,how='left',on='NOC')
medal_winter.sort_values('Total_Medals',ascending=False,inplace=True)
medal_winter=medal_winter[['NOC','Sport','Gold','Silver','Bronze','Total_Medals']]
#medal_winter.head()


# In[92]:


#plot of top 20 countries in total medals in Summer games during 120 years olympic game history 
plt.figure(figsize=(10,10))
plt.subplot(211)
ax=sns.barplot(medal_summer['NOC'].head(20),medal_summer['Total_Medals'].head(20),palette=sns.color_palette("Set3"))
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_xlabel('NOC',fontsize=10)
ax.set_ylabel('Total Medals',fontsize=10)
ax.set_title('Top 20 Countries in Summer Olympics')

#plot of top 20 countries in total medals in Winter games during 120 years olympic game history
plt.subplot(212)
ax=sns.barplot(medal_winter['NOC'].head(20),medal_winter['Total_Medals'].head(20),palette=sns.color_palette("Set3"))
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_xlabel('NOC',fontsize=10)
ax.set_ylabel('Total Medals',fontsize=10)
ax.set_title('Top 20 Countries in Winter Olympics')

plt.subplots_adjust(wspace = 0.4, hspace = 0.8,top = 0.8)
plt.show()


# ### (a). Performances After 1990

# In[50]:


#Summer Games after 1990, Soviet Union-URS does not exisited any more
After1990 = summer[summer.Year>1990]
#After1990.rename(columns={'Sport':'Types of Sports'}, inplace=True)

medal_summer_1990=After1990.groupby(['NOC','Medal'])['Medal'].count()
medal_summer_1990=medal_summer_1990.unstack(level=-1,fill_value=0).reset_index()

medal_summer_1990['Total_Medals']=medal_summer_1990['Bronze']+medal_summer_1990['Gold']+medal_summer_1990['Silver']
Sports_in_total_summer_1990=After1990.groupby('NOC')['Sport'].nunique().to_frame().reset_index()

medal_summer_1990=pd.merge(medal_summer_1990,Sports_in_total_summer_1990,how='left',on='NOC')
medal_summer_1990.sort_values('Total_Medals',ascending=False,inplace=True)
medal_summer_1990=medal_summer_1990[['NOC','Sport','Gold','Silver','Bronze','Total_Medals']]

#Sports_in_total_summer_1990.head()


# In[51]:


#number of medals by each country/team through out olympic game history
medal_summer.head(10)


# In[52]:


#number of medals by each country/team after 1990s.
medal_summer_1990.head(10)


# In[53]:


#plot of top20 countries in summer olympic games after 1990s.
plt.figure(figsize=(10,10))
#plt.subplot(211)
ax=sns.barplot(medal_summer_1990['NOC'].head(20),medal_summer_1990['Total_Medals'].head(20),palette=sns.color_palette('Set2'))
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_xlabel('NOC',fontsize=10)
ax.set_ylabel('Total Medals',fontsize=10)
ax.set_title('Top 20 Countries in Summer Olympics after 1990')
plt.show()


# In[54]:


#plot of top gold holders after 1990s
plt.figure(figsize=(10,10))
#plt.subplot(211)
ax=sns.barplot(medal_summer_1990['NOC'].head(20),medal_summer_1990['Gold'].head(20),palette=sns.color_palette('Set2'))
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_xlabel('NOC',fontsize=10)
ax.set_ylabel('Golds',fontsize=10)
ax.set_title('Top 20 Countries win Golds in Summer Olympics after 1990')
plt.show()


# ### (b). Performance of China

# In[55]:


#popular sports of China
CHN=olympic_data[(olympic_data['NOC']=='CHN')]
CHN_games=CHN.groupby('Sport').size().to_frame().reset_index()
CHN_games.columns=['Sport','Count']
CHN_games.sort_values(by='Count',ascending=False,inplace=True)
CHN_games.head()


# In[57]:


#adept sports of China
CHN=olympic_data[(olympic_data['NOC']=='CHN')& (olympic_data['Medal']=='Gold')]
CHN_gold=CHN.groupby('Sport').size().to_frame().reset_index()
CHN_gold.columns=['Sport','Count']
CHN_gold.sort_values(by='Count',ascending=False,inplace=True)
CHN_gold.head()


# In[87]:


#sports that China wom most medals
plt.figure(figsize=(12,15))
plt.subplot(311)
ax=sns.barplot(CHN_games['Sport'].head(5),CHN_games['Count'].head(5),palette=sns.color_palette('Paired'))
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_title('Sports that China won most medals')

#sports that China wom most golds
plt.subplot(312)
ax1 = sns.barplot(CHN_gold['Sport'].head(5),CHN_gold['Count'].head(5),palette=sns.color_palette('Paired'))
plt.title("Sports that China won most golds")

plt.subplots_adjust(wspace = 0.2, hspace = 1,top = 1)
plt.show()


# In[59]:


#adept events of China
CHN=olympic_data[(olympic_data['NOC']=='CHN')& (olympic_data['Medal']=='Gold')]
CHN_event=CHN.groupby('Event').size().to_frame().reset_index()
CHN_event.columns=['Event','Count']
CHN_event.sort_values(by='Count',ascending=False,inplace=True)
CHN_event.head(10)


# In[60]:


#best athelets in China
CHN_Top1=CHN.groupby(['Name','Sport','Sex']).size().to_frame().reset_index()
CHN_Top1.columns=['Name','Sport','Sex','Golds']
CHN_Top1.sort_values(by='Golds',ascending=False,inplace=True)
CHN_Top1.head(20)


# ### (c). Performance of U.S.A

# In[61]:


#popular sports of USA
USA=olympic_data[(olympic_data['NOC']=='USA')]
USA_games=USA.groupby('Sport').size().to_frame().reset_index()
USA_games.columns=['Sport','Count']
USA_games.sort_values(by='Count',ascending=False,inplace=True)
USA_games.head()


# In[62]:


#adept sports of USA
USA=olympic_data[(olympic_data['NOC']=='USA')& (olympic_data['Medal']=='Gold')]
USA_gold=USA.groupby('Sport').size().to_frame().reset_index()
USA_gold.columns=['Sport','Count']
USA_gold.sort_values(by='Count',ascending=False,inplace=True)
USA_gold.head()


# In[86]:


#sports that USA wom most medals
plt.figure(figsize=(12,15))
plt.subplot(311)
ax=sns.barplot(USA_games['Sport'].head(5),USA_games['Count'].head(5),palette=sns.color_palette('Set2'))
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_title('Sports that USA won most medals')

#sports that USA wom most golds
plt.subplot(312)
ax1 = sns.barplot(USA_gold['Sport'].head(5),USA_gold['Count'].head(5),palette=sns.color_palette('Set2'))
plt.title("Sports that USA won most golds")

plt.subplots_adjust(wspace = 0.2, hspace = 1,top = 1)
plt.show()


# In[64]:


#adept events of USA
USA=olympic_data[(olympic_data['NOC']=='USA')& (olympic_data['Medal']=='Gold')]
USA_event=USA.groupby('Event').size().to_frame().reset_index()
USA_event.columns=['Event','Count']
USA_event.sort_values(by='Count',ascending=False,inplace=True)
USA_event.head(10)


# In[65]:


#best athelets in USA
USA_Top1=USA.groupby(['Name','Sport','Sex']).size().to_frame().reset_index()
USA_Top1.columns=['Name','Sport','Sex','Golds']
USA_Top1.sort_values(by='Golds',ascending=False,inplace=True)
USA_Top1.head(20)


# ### (d). Top1 Athelets in the World

# In[66]:


World_Top1=olympic_data.groupby(['Name','Sport','Sex','NOC']).size().to_frame().reset_index()
World_Top1.columns=['Name','Sport','Sex','NOC','Golds']
World_Top1.sort_values(by='Golds',ascending=False,inplace=True)
World_Top1.head(20)


# ## 9. Compare with GDP

# In[67]:


#read GDP data
GDP = pd.read_csv('API_NY.GDP.MKTP.CD_DS2_en_csv_v2_10515210.csv', skiprows = 4, index_col = 'Country Name')
#Select years that have summer olympic games
GDP=GDP.loc[:,['Country Code','1960','1964','1968','1972','1976','1980','1984','1988','1992',
               '1996','2000','2004','2008','2012','2016']]
#GDP.dropna(how='any', inplace=True)
GDP.rename(columns={'Country Code':'NOC'}, inplace=True)
GDP.dropna(how='any', inplace=True)
GDP.head()


# In[68]:


#reset index of GDP data
GDP=GDP.set_index('NOC')
GDP.head()


# In[69]:


#find GDP of USA
USAGDP = ['USA']
USAGDP = GDP.loc[USAGDP]
USAGDP


# In[70]:


#Transpose USAGDP
USAGDPTransposed = USAGDP.transpose()
#aTransposedPlot = aTransposed.plot.line(legend=False,figsize=(12,8),fontsize=10)
USAGDPTransposed = USAGDP.transpose().reset_index()
USAGDPTransposed.rename(columns={'index':'Year'}, inplace=True)
USAGDPTransposed.rename(columns={'USA':'GDP'}, inplace=True)
usa = USAGDPTransposed
usa.head()


# In[71]:


#find GDP of CHN
CHNGDP = ['CHN']
CHNGDP = GDP.loc[CHNGDP]
CHNGDP


# In[72]:


#Transpose CHNGDP
CHNGDPTransposed = CHNGDP.transpose()
CHNGDPTransposed = CHNGDP.transpose().reset_index()
CHNGDPTransposed.rename(columns={'index':'Year'}, inplace=True)
CHNGDPTransposed.rename(columns={'CHN':'GDP'}, inplace=True)
china = CHNGDPTransposed
#CHNGDPTransposed.head()
china.head()


# ### (a). Compare Number of Golds with GDP

# ### -China

# In[73]:


#China count of golds by year
CHNgolds = CHN.loc[:,['Year','Medal']]
CHNgolds['Counts'] = CHNgolds.groupby(['Year'])['Medal'].transform('count')
CHNgolds.head()
#CHNgolds.info()


# In[78]:


#GDP by year
plt.figure(figsize=(14,10))
plt.subplot(311)
ax=sns.pointplot(x=china['Year'],y=china['GDP'],dodge=True, color='lightcoral')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_title('GDP')

#Count of golds by year
plt.subplot(312)
ax1=sns.barplot(x=CHNgolds['Year'],y=CHNgolds['Counts'],dodge=True, color='gold')
plt.title("Count of Golds")

plt.subplots_adjust(wspace = 0.2, hspace = 1,top = 1)
plt.show()


# ### -USA

# In[75]:


#USA count of golds by year
USAgolds = USA.loc[:,['Year','Medal']]
USAgolds['Counts'] = USAgolds.groupby(['Year'])['Medal'].transform('count')
USAgolds.head()


# In[79]:


#GDP by year
plt.figure(figsize=(16,10))
plt.subplot(311)
ax=sns.pointplot(x=usa['Year'],y=usa['GDP'],dodge=True, color='lightblue')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_title('GDP')

##Count of golds by year
plt.subplot(312)
ax1=sns.barplot(x=USAgolds['Year'],y=USAgolds['Counts'],dodge=True, color='gold')
plt.title("Count of Golds")

plt.subplots_adjust(wspace = 0.2, hspace = 1,top = 1)
plt.show()


# ### (b). Compare Medals in Total with GDP

# In[80]:


#Calculate total medals for each country by Year
medal=olympic_data.groupby(['NOC','Year','Medal'])['Medal'].count()
medal=medal.unstack(level=-1,fill_value=0).reset_index()

medal['MedalInTotal'] = medal['Bronze']+medal['Gold']+medal['Silver']
medal.head()


# ### -China

# In[81]:


chinaTotal=medal[(medal['NOC']=='CHN')]
chinaTotal.head()


# In[82]:


#GDP by year
plt.figure(figsize=(16,10))
plt.subplot(311)
ax=sns.pointplot(x=china['Year'],y=china['GDP'],dodge=True, color='lightcoral')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_title('GDP')

##Count of golds by year
plt.subplot(312)
ax1=sns.barplot(x=chinaTotal['Year'],y=chinaTotal['MedalInTotal'],dodge=True, color='orange')
plt.title("Medals In Total")

plt.subplots_adjust(wspace = 0.2, hspace = 1,top = 1)
plt.show()


# ### -USA

# In[83]:


usaTotal=medal[(medal['NOC']=='USA')]
usaTotal.head()


# In[85]:


#GDP by year
plt.figure(figsize=(16,10))
plt.subplot(311)
ax=sns.pointplot(x=usa['Year'],y=usa['GDP'],dodge=True, color='lightblue')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_title('GDP')

##Count of golds by year
plt.subplot(312)
ax1=sns.barplot(x=usaTotal['Year'],y=usaTotal['MedalInTotal'],dodge=True, color='orange')
plt.title("Medals In Total")

plt.subplots_adjust(wspace = 0.2, hspace = 1,top = 1)
plt.show()


# ## 10. Export Data as CSV File

# In[ ]:


##export olympic_data to csv file for further use
olympic_data.to_csv('olympic_data.csv',index=False)

