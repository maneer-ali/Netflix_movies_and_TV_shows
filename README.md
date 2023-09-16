# Project Name - Netflix Movies and TV Show Clustering
# Project Type - Unsupervised Machine Learning
# Contribution - Individual
# Name - - Maneer Ali
# Project Summary
The aim of project is to analyze the Netflix Dataset and TV shows untill 2019, sourced from the third-party search engine Fixable. The goal is to group the content into relevant clusters using NLP techniques to improve the user experience through a recommendation system. This will help prevent subscriber churn for Netflix, which country has over 220 million subscribers.

Additionally, the dataset will be analyzed to uncover insights and trends in the streaming entertainment industry.

The project followed a step-by-step process:

Handling null values in the dataset.
Managing nested columns (director, cast, listed_in, country) for better visualization.
Binning the rating attribute into categories (adult, children's, family-friendly, not-rated).
Performing Exploratory Data Analysis (EDA) to gain insights for preventing subscriber churn.
Creating clusters using attributes like director, cast, country, genre, rating and description. These attributes were tokenized, preprocessed and vectorized using TF-IDF vectorizer.
Reducing the dimensionality of the dataset using PCA to improve performance.
Employing K-means Clustering and Agglomerative Hierarchical Clustering algorithms, determining optimal cluster numbers (4 for K-means, 2 for hierarchical clustering) through various evaluation methods.
Developing a content-based recommender system using cosine similarity matrix to provide personalized recommendations to users and reduce subscriber churn for Netflix.
This comprehensive analysis and recommendation system are expected to enhance user satisfaction, leading to improved retentation rates for Netflix.

# Let's Begin !
# 1. Know Your Data
## Import Libraries

```
# Import Libraries
## Data Maipulation Libraries
import numpy as np
import pandas as pd
import datetime as dt

## Data Visualisation Libraray
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns
%matplotlib inline
import plotly.graph_objects as go


# libraries used to process textual data
import string
string.punctuation
import nltk
nltk.download('punkt')
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# libraries used to implement clusters
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram

# Library of warnings would assist in ignoring warnings issued
import warnings;warnings.filterwarnings('ignore')
import warnings;warnings.simplefilter('ignore')
```

# Dataset Loading

```
# Mounting drive
from google.colab import drive
drive.mount('/content/drive')

# Load Dataset
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Module 1/NETFLIX MOVIES AND TV SHOWS CLUSTERING.csv')
```

# Dataset First Look
```
df.head()
```

# Dataset Rows & Columns count
```
# Dataset Rows & Columns count
print(f"Rows and Column count in the Dataset: Rows= {df.shape[0]}, Columns= {df.shape[1]}")
```

# Dataset Information
```
# Dataset Info
df.info()
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7787 entries, 0 to 7786
Data columns (total 12 columns):
 #   Column        Non-Null Count  Dtype 
---  ------        --------------  ----- 
 0   show_id       7787 non-null   object
 1   type          7787 non-null   object
 2   title         7787 non-null   object
 3   director      5398 non-null   object
 4   cast          7069 non-null   object
 5   country       7280 non-null   object
 6   date_added    7777 non-null   object
 7   release_year  7787 non-null   int64 
 8   rating        7780 non-null   object
 9   duration      7787 non-null   object
 10  listed_in     7787 non-null   object
 11  description   7787 non-null   object
dtypes: int64(1), object(11)
memory usage: 730.2+ KB

# Duplicate Values

```
# Dataset Duplicate Value Count
print(f"The total number of duplicated observations in the dataset: {df.duplicated().sum()}")
```
The total number of duplicated observations in the dataset: 0
It's good to see that we do not have any duplicated observation in our dataset.

# Missing values/Null Values

```
# Missing Values/Null Values Count
print("-"*50)
print("Null value count in each of the variable: ")
print("-"*50)
print(df.isna().sum())
print("-"*50)

# Percentage of null values in each category
print("Percentage of null values in each variable: ")
print("-"*50)
null_count_by_variable = df.isnull().sum()/len(df)
print(f"{null_count_by_variable*100}%")
print("-"*50)
```
--------------------------------------------------
Null value count in each of the variable: 
--------------------------------------------------
show_id            0
type               0
title              0
director        2389
cast             718
country          507
date_added        10
release_year       0
rating             7
duration           0
listed_in          0
description        0
dtype: int64
--------------------------------------------------
Percentage of null values in each variable: 
--------------------------------------------------
show_id          0.000000
type             0.000000
title            0.000000
director        30.679337
cast             9.220496
country          6.510851
date_added       0.128419
release_year     0.000000
rating           0.089893
duration         0.000000
listed_in        0.000000
description      0.000000
dtype: float64%
--------------------------------------------------

```
# Visualizing the missing values
# Checking Null Value by plotting Heatmap
plt.figure(figsize=(7,5))
sns.heatmap(df.isnull(), cbar=True)
plt.show()
```

```
# Visualizing the missing values
plt.figure(figsize=(15,8))
plots= sns.barplot(x=df.columns,y=df.isna().sum())
plt.grid(linestyle='--', linewidth=0.3)

for bar in plots.patches:
      plots.annotate(bar.get_height(),
                     (bar.get_x() + bar.get_width() / 2,
                      bar.get_height()), ha='center', va='center',
                     size=12, xytext=(0, 8),
                     textcoords='offset points')
plt.show()
```


## What did you know about your dataset?
The dataset "Netflix Movies and TV Shows Clustering" comprises 12 colimns, with only one column having an integer data type. It does not contain any duplicate values, but it does not have null values in five columns: director, cast, country, date_added and rating.

This dataset provides a valuable resource for exploring trends in the range of movies nad TV shows available on netflix. Additionally, it can be utilized for developing clustering models to categorize similar titles together based on shared such as genre, countryof origin and rating.

# 2. Understanding Your Variables

```
# Dataset Columns
print(f"Available columns:\n{df.columns.to_list()}")
```
Available columns:
['show_id', 'type', 'title', 'director', 'cast', 'country', 'date_added', 'release_year', 'rating', 'duration', 'listed_in', 'description']

```
# dataset Describe
df.describe(include='all').T
```

# Variable Description
The variable description of the Netflix Movies and TV Shows Clustering Dataset is as follows:

**show_id:** Unique for each movie/show.
**type:** Indicates whether the entry is a movie or a TV show.
**title:** Name of the movie or TV show
**director:** Name of the director(s) of the movie or TV show
**cast:** Names of the actors and acteresses featured in the movie or TV show.
**country:** Country or countries where the movie or TV show was produced.
**date_added:** Date when the movie or TV show was added to Netflix.
**release_year:** Year when the movie or TV show was released.
**rating:** TV rating or movie rating of the movie or TV show.
**duration:** Length of the movie or TV show in minutes or seasons.
**listed_in:** categories or games of the movie or TV show.
**description:** Brief synopsis or summary of the movie or TV show.

## Check Unique Values for each variable.

```
# Check Unique Values for each variable.
print(f"The number of unique values in: ")
print("-"*35)
for i in df.columns:
  print(f"'{i}' : {df[i].nunique()}")
```

The number of unique values in: 
-----------------------------------
'show_id' : 7787
'type' : 2
'title' : 7787
'director' : 4049
'cast' : 6831
'country' : 681
'date_added' : 1565
'release_year' : 73
'rating' : 14
'duration' : 216
'listed_in' : 492
'description' : 7769
# 3. Data Wrangling
## Data Wrangling Code
# 1. Handling Null values from each feature
```
# Missing Values/Null Values Count
print("-"*50)
print("Null value count in each of the variable: ")
print("-"*50)
print(df.isna().sum())
print("-"*50)

# Let's find out the percentage of null values in each category in order to deal with it.
print("Percentage of null values in each variable: ")
print("-"*50)
```

--------------------------------------------------
Null value count in each of the variable: 
--------------------------------------------------
show_id            0
type               0
title              0
director        2389
cast             718
country          507
date_added        10
release_year       0
rating             7
duration           0
listed_in          0
description        0
dtype: int64
--------------------------------------------------
Percentage of null values in each variable: 
--------------------------------------------------
show_id          0.000000
type             0.000000
title            0.000000
director        30.679337
cast             9.220496
country          6.510851
date_added       0.128419
release_year     0.000000
rating           0.089893
duration         0.000000
listed_in        0.000000
description      0.000000
dtype: float64%
--------------------------------------------------
```
df["date_added"].value_counts()
```

January 1, 2020      118
November 1, 2019      94
December 31, 2019     76
March 1, 2018         76
October 1, 2018       72
                    ... 
October 12, 2014       1
March 22, 2020         1
March 31, 2013         1
December 12, 2019      1
January 11, 2020       1
Name: date_added, Length: 1565, dtype: int64

```
df['rating'].value_counts()
```

TV-MA       2863
TV-14       1931
TV-PG        806
R            665
PG-13        386
TV-Y         280
TV-Y7        271
PG           247
TV-G         194
NR            84
G             39
TV-Y7-FV       6
UR             5
NC-17          3
Name: rating, dtype: int64

```
df['country'].value_counts()
```

United States                                                   2555
India                                                            923
United Kingdom                                                   397
Japan                                                            226
South Korea                                                      183
                                                                ... 
Russia, United States, China                                       1
Italy, Switzerland, France, Germany                                1
United States, United Kingdom, Canada                              1
United States, United Kingdom, Japan                               1
Sweden, Czech Republic, United Kingdom, Denmark, Netherlands       1
Name: country, Length: 681, dtype: int64
1. Since 'date_added' and rating has very less percentage of null count so we can drop those observations to avaoid any biasness in our clustering model.
2. We cannot drop or impute any values in 'director' and 'cast' as the null hypothesis is comparitevely high and we do not know date of those actual movie/TV shows, so its better to replace those entries with 'unknown'.
3. We can fill null values of 'country' with mode as we only have 6% null values and most of the movies/shows are from US only.

```
## Imputing null value as per our discussion
# imputing with unknown in null values of director and cast feature
df[['director','cast']]=df[['director','cast']].fillna("Unknown")

# Imputing null values of country with Mode
df['country']=df['country'].fillna(df['country'].mode()[0])

# Dropping remaining null values of date_added and rating
df.dropna(axis=0, inplace=True)

# Rechecking the Missing Values/Null Values Count
print("-"*50)
print("Null value count in each of the variable: ")
print("-"*50)
print(df.isna().sum())
print("-"*50)

# Rechecking the percentage of null values in each category
print("Percentage of null values in each variable: ")
print("-"*50)
null_count_by_variable = df.isnull().sum()/len(df)
print(f"{null_count_by_variable*100}%")
print("-"*50)
```
--------------------------------------------------
Null value count in each of the variable: 
--------------------------------------------------
show_id         0
type            0
title           0
director        0
cast            0
country         0
date_added      0
release_year    0
rating          0
duration        0
listed_in       0
description     0
dtype: int64
--------------------------------------------------
Percentage of null values in each variable: 
--------------------------------------------------
show_id         0.0
type            0.0
title           0.0
director        0.0
cast            0.0
country         0.0
date_added      0.0
release_year    0.0
rating          0.0
duration        0.0
listed_in       0.0
description     0.0
dtype: float64%
--------------------------------------------------

```
# Let's create a copy of dataframe and unnest the original one
df_new= df.copy()

# Unnesting 'Directors' column
dir_constraint=df['director'].apply(lambda x: str(x).split(', ')).tolist()
df1 = pd.DataFrame(dir_constraint, index = df['title'])
df1 = df1.stack()
df1 = pd.DataFrame(df1.reset_index())
df1.rename(columns={0:'Directors'},inplace=True)
df1 = df1.drop(['level_1'],axis=1)
df1.sample(10)

# Unnesting 'Cast' column
cast_constraint=df['cast'].apply(lambda x: str(x).split(', ')).tolist()
df2 = pd.DataFrame(cast_constraint, index = df['title'])
df2 = df2.stack()
df2 = pd.DataFrame(df2.reset_index())
df2.rename(columns={0:'Actors'},inplace=True)
df2 = df2.drop(['level_1'],axis=1)
df2.sample(10)

# Unnesting 'listed_in' column
listed_constraint=df['listed_in'].apply(lambda x: str(x).split(', ')).tolist()
df3 = pd.DataFrame(listed_constraint, index = df['title'])
df3 = df3.stack()
df3 = pd.DataFrame(df3.reset_index())
df3.rename(columns={0:'Genre'},inplace=True)
df3 = df3.drop(['level_1'],axis=1)
df3.sample(10)

# Unnesting 'country' column
country_constraint=df['country'].apply(lambda x: str(x).split(', ')).tolist()
df4 = pd.DataFrame(country_constraint, index = df['title'])
df4 = df4.stack()
df4 = pd.DataFrame(df4.reset_index())
df4.rename(columns={0:'Country'},inplace=True)
df4 = df4.drop(['level_1'],axis=1)
df4.sample(10)
```

**Great, we have successfully seperated the nested columns. Now let's just merge all the created dataframe into the single merged dataframes.***

```
## Merging all the unnested dataframes
# Merging director and cast
df5 = df2.merge(df1,on=['title'],how='inner')

# Merging listed_in with merged of (director and cast)
df6 = df5.merge(df3,on=['title'],how='inner')

# Merging country with merged of [listed_in with merged of (director and cast)]
df7 = df6.merge(df4,on=['title'],how='inner')

# Head of final merged dataframe
df7.head()
```

**Cool, now let's merge this dataframe with the original one on the left join to avoid information loss.**

```
# Merging unnested data with the created dataframe in order to make the final dataframe
df = df7.merge(df[['type', 'title', 'date_added', 'release_year', 'rating', 'duration','description']],on=['title'],how='left')
df.head()
```

# 3. Typecasting of attributes

```
# Checking info of the dataset before typecasting
df.info()
```
<class 'pandas.core.frame.DataFrame'>
Int64Index: 175807 entries, 0 to 175806
Data columns (total 11 columns):
 #   Column        Non-Null Count   Dtype 
---  ------        --------------   ----- 
 0   title         175807 non-null  object
 1   Actors        175807 non-null  object
 2   Directors     175807 non-null  object
 3   Genre         175807 non-null  object
 4   Country       175807 non-null  object
 5   type          175807 non-null  object
 6   date_added    175807 non-null  object
 7   release_year  175807 non-null  int64 
 8   rating        175807 non-null  object
 9   duration      175807 non-null  object
 10  description   175807 non-null  object
dtypes: int64(1), object(10)
memory usage: 16.1+ MB

```
# Typecasting duration into integer by removing 'min' and 'season' from the end
df['duration']= df['duration'].apply(lambda x: int(x.split()[0]))

# Typecasting string object to datetime object of date_added column
df['date_added']= pd.to_datetime(df['date_added'])

# Extracting date, day, month and year from date_added column
df["day_added"]= df["date_added"].dt.day
df["month_added"]= df["date_added"].dt.month
df["year_added"]= df["date_added"].dt.year

# Checking info of the dataset after typecating
df.info()
```

<class 'pandas.core.frame.DataFrame'>
Int64Index: 175807 entries, 0 to 175806
Data columns (total 13 columns):
 #   Column        Non-Null Count   Dtype 
---  ------        --------------   ----- 
 0   title         175807 non-null  object
 1   Actors        175807 non-null  object
 2   Directors     175807 non-null  object
 3   Genre         175807 non-null  object
 4   Country       175807 non-null  object
 5   type          175807 non-null  object
 6   release_year  175807 non-null  int64 
 7   rating        175807 non-null  object
 8   duration      175807 non-null  int64 
 9   description   175807 non-null  object
 10  day_added     175807 non-null  int64 
 11  month_added   175807 non-null  int64 
 12  year_added    175807 non-null  int64 
dtypes: int64(5), object(8)
memory usage: 18.8+ MB

# 4. Binning of Rating attribute
In rating columns we have different categories these are content rating classifications that are commonly used in the United States and other countries to indicate the appropriateness of media content for different age groups. Let's understand each of them and binning them accordingly:

**TV-MA:** This rating is used for mature audiences only, and it may contain strong language, violence, nudity and sexual content.

**R:** This rating is used for movies that are intended for audiences 17 and older. It may contain graphic violence, strong language, drug use and sexual content.

**PG-13:** This rating is used for movies that may be suitable for children under 13. It may contain violences, mild to moderate language and suggestive content.

**TV-14:** This rating is used for TV shows that may not be suitable for children under 14. It mau contain violence, strong language, sexual situations and suggestive dialogue.

**TV-PG:** This rating is used for TV shows that may not be suitable for children under 8. It may contain mild violence, language and suggestive content.

**NR:** This stands for "Not Rated". It means that the content has not been rated by a rating board and it may contain material taht is not suitable for all audiences.

**TV-G:** This rating is used for Tv shows that are suitable for all ages. It may contain some mild violence, language and suggestive content.

**TV-Y:** This rating is used for children's TV shows that are suitable for all ages. It is intended to be appropriate for preschool children.

**TV-Y7:** This rating is used for children's TV shows that may not be suitable for children under 7. it may contain mild violence and scary content.

**PG:** This rating is used for movies that may not be suitable for children under 10. It may contain mild language some violence and some suggestive content.

**G:** This rating is used for movies that are suitable for general audiences. It may contain some mild language and some violence.

**NC-17:** This rating is used for movies that are intended for adults only. It may contain explicit sexual content, violence and language.

**TV-Y7-FV:** This rating is used for children's TV show that may not be suitable for children's under 7. It may contain fantasy violence.

**UR:** This stands for "Unrated". It means that the content has not been rated by a rating board and it may contain material that is not suitable for all audiences.

**Lets not complicate it and create bins as following:**

* Adult Content: TV-MA, NC-17, R
* Children Content: TV-PG, PG, TV-G, G
* Teen Content: PG-13, TV-14
* Family-friendly Content: TV-Y, TV-Y7, TV-Y7-FV
* Not Rated: NR, UR

```
# Binning the values in the rating column
rating_map = {'TV-MA':'Adult Content',
              'R':'Adult Content',
              'PG-13':'Teen Content',
              'TV-14':'Teen Content',
              'TV-PG':'Children Content',
              'NR':'Not Rated',
              'TV-G':'Children Content',
              'TV-Y':'Family-friendly Content',
              'TV-Y7':'Family-friendly Content',

```

array(['Adult Content', 'Teen Content', 'Children Content', 'Not Rated',
       'Family-freindly Content'], dtype=object)

```
# Checking head after binning
df.head()
```

# 5. Sepearating the dataframes for further analysis

```
# Spearating the dataframes for further analysis
df_movies= df[df['type']== 'Movie']
df_tvshows= df[df['type']== 'TV Show']

# Printing the shape
print(df_movies.shape, df_tvshows.shape)
```

(126079, 13) (49728, 13)

# What all manipulations have you done and insights you found?
We have divided data wrangling into five different sections:

In this section we have imputed/drop the null values of:

Imputed 'director' and 'cast' with 'Unknown'.
Imputed 'country' with Mode.
Drop null values of 'date_added' and 'rating'(less percentage).
We have unnested values from following features:

* 'director'
* 'cast'
* 'listed_in'
* 'country' We have unnested the values and stored in different dataframes and then merged all the dataframe with the original one using left join in order to get the isolated values of each of the feature.
We have typecasted the following feature:

* 'duration' into integer (Removing min and seasons from the values).
* 'date_added' to datetime (into the required format).
We have also extracted the following features:

* 'date' from 'date_added'.
* 'month' from 'date_added'.
* 'year' from 'date_added'.
We have seen that the 'rating' column contains various coded categories, so we have decided to create 5 bins and distribute the values accordingly.

* Adult: R, UR
* Resticted: TV-PG, PG, TV-G, G
* Teen: PG-13, TV-14
* All Ages: TV-G, TV-Y, TV-Y7, TV-Y7-FV, PG, G, TV-PG
* Not Rated: NR

Lastly we have splited the dataframe into two df one is 'df_movies' that contains only Movies and the other is 'df_tvshows' that contains only TV Shows for our further analysis.

# 4. Data Visualization, Storytelling & Experimenting with charts: Understand the relationships between variables
Chart - 1 (The relative percentage of total number of Movies and TV Shows over Netflix?)

```
# Chart - 1 visualization code

labels = ['TV Show', 'Movie']
values = [df.type.value_counts()[1], df.type.value_counts()[0]]

# Colors
colors = ['#ffd700', '#008000']

# Create pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)])

# Customize layout
fig.update_layout(
    title_text='Type of Content Watched on Netflix',
    title_x=0.5,
    height=500,
    width=500,
    legend=dict(x=0.9),
    annotations=[dict(text='Type of Content', font_size=20, showarrow=False)]
)

# Set colors
fig.update_traces(marker=dict(colors=colors))
```

## 1. Why did you pick the specific chart?
This graph shows us the percent of TV shows and movie data present on Netflix Data set

## 2. What is/are the insights(s) found from the chart?
We can see that the majority of the content on Netflix is movies, which account for around two-thirds of the total content. TV shows make up the remaining one-third of the content.
We can conclude that in the given Data set only 28.3% are TV Shows and 71.7% are Movies.
## 3. Will the gained insights help creating a positive business imapct?
Are there any insights that lead to negative growth? Justify with the specific reason.

Yes! tge production house more focuses on quality movies because there is high competition in the market.
TV Shows are less in numbers hence good opportunity for business.
## Chart - 2(How content is distributed over Netflix?)

```
plt.figure(figsize=(25,10))
for i,j,k in ((df, 'Overall',0),(df_movies, 'Movies',1),(df_tvshows, 'TV Shows',2)):
  plt.subplot(1,3,k+1)
  count= i['rating'].value_counts()
  plt.pie(count, labels=count.index,explode=(0,0,0,0,0.5),colors=['orangered','dodgerblue','lightgreen','mediumslateblue','yellow'],
          autopct='%1.1f%%', labeldistance=1.1,wedgeprops={"edgecolor" : "black", 'linewidth': 1,'antialiased': True})
  plt.title(f"Distribution of Content rating on Netflix '{j}'")
  plt.axis('equal')
plt.show()
```

## 1. Why did you pick the specific chart?
We have choosen this chart to know the percentage of type of content present in the Netflix.

## 2. What is/are the insight(s) found from the chart?
We found that most of the content present in the Netflix belongs to Adult and the teen categories.
Another important insight we can see that Family freindly content less in Movies compatred to TV Shows.
## 3. Will the gained insightshelp creating a positive busniness impact?
Are there any insights that lead to negative growth? Justify with specific reason.

For high gains production house should focus on Teen and Adult content.
There is good chances of growth in family-friendly category in TV Shows
## Chart -3 (Who are the top actors performing in Movies and TV Shows?)

```
# Top 10 casts in Movies and TV Shows
plt.style.use('default')
plt.figure(figsize=(23,8))
for i,j,k in ((df_movies, 'Movies',0),(df_tvshows, 'TV Shows',1)):
  plt.subplot(1,2,k+1)
  df_actor = i.groupby(['Actors']).agg({'title':'nunique'}).reset_index().sort_values(by=['title'],ascending=False)[1:10]
  plots= sns.barplot(y = "Actors",x = 'title', data = df_actor, palette='Set1')
  plt.title(f'Actors appeared in most of the {j}')
  plt.grid(linestyle='--', linewidth=0.3)
  plots.bar_label(plots.containers[0])
plt.show()
```

## 1. Why did you pick the specific chart?
To know which actors are more popular on Netflix.

## 2. What is/are the insight(s) found from the chart?
We found an interseting insight that most of the Actors in Movies are from INDIA.
No popular actors from India in TV Shows.
## 3. Will the gained insightshelp creating a positive busniness impact?
Are there any insights that lead to negative growth? Justify with specific reason.

Indians are movie lover, they love to watch movies hence business should target Indian audience for Movies.
## Chart - 4(Who are the top Directors directing Movies and TV Shows?)

```
# Top 10 Directors in Movies and TV Shows
plt.figure(figsize=(23,8))
for i,j,k in ((df_movies, 'Movies',0),(df_tvshows, 'TV Shows',1)):
  plt.subplot(1,2,k+1)
  df_director = i.groupby(['Directors']).agg({'title':'nunique'}).reset_index().sort_values(by=['title'],ascending=False)[1:10]
  plots= sns.barplot(y = "Directors",x = 'title', data = df_director, palette='Paired')
  plt.title(f'Directors appeared in most of the {j}')
  plt.grid(linestyle='--', linewidth=0.3)
  plots.bar_label(plots.containers[0])
plt.show()
```

## 1. Why did you pick the specific chart?
To know which director is popular in Movies and which one is popular in TV Shows.

## 2. What is/are the insight(s) found from the chart?
We found that most of the movies directed by jan suter.
Most TV Shows firected by ken burns.
## 3. Will the gained insightshelp creating a positive busniness impact?
Are there any insights that lead to negative growth? Justify with specific reason.

Movies/TV Shows producers can select the popular director for their upcoming projects.

## Chart - 5 (What are the top 10 Countries involved in content creation?)

```
df_country = df.groupby(['Country']).agg({'title': 'nunique'}).reset_index().sort_values(by=['title'],ascending=False)[:10]
plt.figure(figsize=(15,6))
plots= sns.barplot(y = "Country",x = 'title', data = df_country)
plt.xticks(rotation = 60)
plt.title('Top 10 Countries for content creation')
plt.grid(linestyle='--', linewidth=0.3)
plots.bar_label(plots.containers[0])
plt.show()
```

## 1. Why did you pick the specific chart?
To know which country produces Maximum number of TV Shows and Movies.

## 2. What is/are the insight(s) found from the chart?
The United States is the top country producing both movies and TV Shows o Netflix. This suggets that Netflix is heavily influnced by American content.
India is the second-highest producer of movies on Netflix, indicating the growing popularity of Bollywood movies worldwide.
Country like Canada, France, Japan also have significant presence in the data set showing diversity of content on the Netflix.
# 3. Will the gained insightshelp creating a positive busniness impact?
Are there any insights that lead to negative growth? Justify with specific reason.

Yes, the insights gained can have a positive impact on Netflix's business by highlighting opportunities for growth and expansion, such as investing in American and Bollywood content and acquring more diverse content.

## Chart - 6 (Which Countries has the highest spread of Movies and TV Shows over Netflix)

```
# Analysing top 15 countries with the most content
plt.figure(figsize=(18,5))
plt.grid(linestyle='--', linewidth=0.3)
sns.countplot(x=df['Country'],order=df['Country'].value_counts().index[0:15],hue=df['type'],palette="Set1")
plt.xticks(rotation=50)
plt.title('Top 15 countries with most contents', fontsize=15, fontweight='bold')
plt.show()

plt.figure(figsize=(20,8))
for i,j,k in ((df_movies, 'Movies',0),(df_tvshows, 'TV Shows',1)):
  plt.subplot(1,2,k+1)
  df_country = i.groupby(['Country']).agg({'title':'nunique'}).reset_index().sort_values(by=['title'],ascending=False)[:10]
  plots= sns.barplot(y= "Country",x = 'title', data = df_country, palette='Set1')
  plt.title(f'Top 10 Countries launching {j} back to back')
  plt.grid(linestyle='--', linewidth=0.3)
  plots.bar_label(plots.containers[0])
plt.show()
```

## 1. Why did you pick the specific chart?
To know which country produces which type of content the most.

## 2. What is/are the insight(s) found from the chart?
India produces most amount of Movies in comaprision to TV Shows.
Japan and South Korea produces more TV Shows in comparision to Movies
## 3. Will the gained insightshelp creating a positive busniness impact?
Are there any insights that lead to negative growth? Justify with specific reason.

Yes, the insights gained can have a positive impact on Netflix's business by highlighting opportunities for growth and expansion, such asacquring and producing more movies from India and more TV Shows from Japan and South Korea.

## Chart - 7(Which Genres are popular in Netflix?)
```
plt.figure(figsize=(23,8))
df_genre = df.groupby(['Genre']).agg({'title': 'nunique'}).reset_index().sort_values(by=['title'],ascending=False)[:10]
plots= sns.barplot(y = "Genre",x = 'title', data = df_genre)
plt.title(f'Most popular genre on Netflix')
plt.grid(linestyle='--', linewidth=0.3)
plots.bar_label(plots.containers[0])
plt.show()

plt.figure(figsize=(23,8))
for i,j,k in ((df_movies, 'Movies',0),(df_tvshows, 'TV Shows',1)):
  plt.subplot(1,2,k+1)
  df_genre = i.groupby(['Genre']).agg({'title':'nunique'}).reset_index().sort_values(by=['title'],ascending=False)[:10]
  plots= sns.barplot(y= "Genre",x = 'title', data = df_genre, palette='Set1')
  plt.title(f'Top 10 Countries launching {j} back to back')
  plt.grid(linestyle='--', linewidth=0.3)
  plots.bar_label(plots.containers[0])
  plt.yticks(rotation = 45)
plt.show()
```

## 1. Why did you pick the specific chart?
This graph tells us which genre is most popular in Netflix

## 2. What is/are the insight(s) found from the chart?
International movies genre is most popular in both TV Shows and Movies category. Followed by Drama and comedy.
## 3. Will the gained insightshelp creating a positive busniness impact?
Are there any insights that lead to negative growth? Justify with specific reason.

The insights gained can have a positive impact on Netflix's business by helping the paltform understand what genres and types of content are popular with its audience. Thsi information can help Netflix tailor its content acquisition and production strategies to better cater to the preferences of its viewers, which can lead to increased engagement and customer staisfaction.

## Chart - 8 (Total number of Movies/TV Shows released and added per year on Netflix?)

```
plt.figure(figsize=(20,6))
for i,j,k in ((df_movies, 'Movies',0),(df_tvshows, 'TV Shows',1)):
  plt.subplot(1,2,k+1)
  df_release_year = i.groupby(['release_year']).agg({'title':'nunique'}).reset_index().sort_values(by=['release_year'],ascending=False)[:14]
  plots= sns.barplot(x= "release_year",y = 'title', data = df_release_year, palette='husl')
  plt.title(f'{j} released by year')
  plt.ylabel(f"Number of {j} released")
  plt.grid(linestyle='--', linewidth=0.3)

  for bar in plots.patches:
    plots.annotate(bar.get_height(),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=12, xytext=(0,8),
                   textcoords='offset points')
plt.show()

plt.figure(figsize=(20,6))
for i,j,k in ((df_movies, 'Movies',0),(df_tvshows, 'TV Shows',1)):
  plt.subplot(1,2,k+1)
  df_country = i.groupby(['year_added']).agg({'title':'nunique'}).reset_index().sort_values(by=['year_added'],ascending=False)
  plots= sns.barplot(x= 'year_added',y = 'title', data = df_country, palette='husl')
  plt.title(f'{j} added to Netflix by year')
  plt.ylabel(f"Number of {j} added on Netflix")
  plt.grid(linestyle='--', linewidth=0.3)

  for bar in plots.patches:
    plots.annotate(bar.get_height(),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                    size=12, xytext=(0, 8),
                    textcoords='offset points')
plt.show()
```

## 1. Why did you pick the specific chart?
This graph shows us how many movies and TV Show released and added in a year on Netflix.

## 2. What is/are the insight(s) found from the chart?
We can see that the number of movies and TV shows added on Netflix has been increasing srteadily every year. But since 2018, the number of Movies released on Netflix has been lowered and the number of TV shows has been significantly increased. In terms of movies and TV Shows addition, in 2020 number of Movies added as compared to 2019 were very less on the other side number of TV show were more as compared to 2019.

## 3. Will the gained insightshelp creating a positive busniness impact?
Are there any insights that lead to negative growth? Justify with specific reason.

The insights that the number of movies added has decreased since 2018 while the number of TV shows added has significantly increased could potentially lead to negative growth for Netflix. This could be due to various reasons such as changing consumer preferences, increased competition from other streaming services and higher production costs associated with creating movies.

To mitigate the potential negative impact, Netflix could explore strategies to diversify its content offerings and adapt to changing consumer preferences. This could include investing in a mix of movies. TV Shows and other forms of original content such as documentations, limited series and stand-up comedy specials. By diversifying its content offerings, Netflix can attract a wider audience and maintain its relevance in the ever-evolving streaming landscape.

## Chart - 9(Total Number of Movies/TV Shows added per month on Netflix)

```
plt.figure(figsize=(23,8))
for i,j,k in ((df_movies, 'Movies',0),(df_tvshows, 'TV Shows',1)):
  plt.subplot(1,2,k+1)
  df_month = i.groupby(['month_added']).agg({'title':'nunique'}).reset_index().sort_values(by=['month_added'],ascending=False)
  plots= sns.barplot(x= 'month_added',y = 'title', data = df_month, palette='husl')
  plt.title(f'{j} added to Netflix by month')
  plt.ylabel(f"Number of {j} added on Netflix")
  plt.grid(linestyle='--', linewidth=0.3)

  for bar in plots.patches:
    plots.annotate(bar.get_height(),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                    size=12, xytext=(0, 8),
                    textcoords='offset points')
plt.show()
```

## 1. Why did you pick the specific chart?
We have plotted this graph to know in which month the movie/tvshows are added is maximum and in which year minimum.

## 2. What is/are the insight(s) found from the chart?
We found that October, November and December are the most popular months for TV shows addition.
January, October and December are the most popular months for movie addition.
February is the least popular month for the movies and TV shows to be added on Netflix.
## 3. Will the gained insightshelp creating a positive busniness impact?
Are there any insights that lead to negative growth? Justify with specific reason.

The insights gained can help Netflix create a positive business impact by identifying the most popular months for new content addition. This can help Netflix plan content releases during peak periods, leading to increased user engagement and relation.

The insigh that February is the most popular month for new content additions could potentially lead to negative growth if Netflix does not maintain a consistent flow of new content during this period. It is important for Netflix to keep its audience engaged throughout the year to avoid disstaisfaction and potential loss of subscribers.

## Chart - 10(Total Number of Movies/TV Shows added per day on Netflix)

```
plt.figure(figsize=(23,8))
for i,j,k in ((df_movies, 'Movies',0),(df_tvshows, 'TV Shows',1)):
  plt.subplot(1,2,k+1)
  df_day = i.groupby(['day_added']).agg({'title':'nunique'}).reset_index().sort_values(by=['day_added'],ascending=False)
  plots= sns.barplot(x= 'day_added',y = 'title', data = df_day, palette='husl')
  plt.title(f'{j} added to Netflix by day')
  plt.ylabel(f"Number of {j} added on Netflix")
  plt.grid(linestyle='--', linewidth=0.3)

  for bar in plots.patches:
    plots.annotate(bar.get_height(),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                    size=12, xytext=(0, 8),
                    textcoords='offset points', rotation=90)
plt.show()
```

## 1. Why did you pick the specific chart?
The graph shows us the day when most of the movies added in a month.:

## 2. What is/are the insight(s) found from the chart?
From the above bar plots, it can be observed that most of the movies and TV shows are added at the beginning or middle of the month. It could be because most people tend to have more free time at the beginning of the month after getting paid and releasing new content during that time could increase viwership. By releasing new content at the beginning and middle of the month, subscribers are more likely to feel that they are getting value for their money, which could lead to increased retention rates.

## 3. Will the gained insightshelp creating a positive busniness impact?
Are there any insights that lead to negative growth? Justify with specific reason.

Yes, releasing new content at regular intervals helps to keep users engaged with the platform, as they will have something new to look forward to every few weeks. This can lead to increased viewing hours and user satisfaction, both of which can have positive impacts on the business.

## Chart - 11 (What is the Month-wise number of content added in each year on Metflix?)

```
plt.figure(figsize=(20,8))
df_year_month = df.groupby(['year_added','month_added']).agg({'title':'nunique'}).reset_index().sort_values(by=['year_added'],ascending=False)
sns.lineplot(x = 'year_added',y='title', data = df_year_month, palette = 'hls', hue=df_year_month['month_added'], marker='o')
plt.grid(linestyle='--', linewidth=0.3)
plt.show()
```

## 1. Why did you pick the specific chart?
The bivariate graph helps us in knowing which month is dominating in adding movie/tvshows in a year.

## 2. What is/are the insight(s) found from the chart?
We can see that there is no specific trend is followed, insted of this some consecutive years shows month wise trend.
From 2008 to 2009 we see movies added in the month of February and from 2009 to 2011 movies added in the month of February and October.
After 2015 majority content added in the month of coctober to december.
## 3. Will the gained insightshelp creating a positive busniness impact?
Are there any insights that lead to negative growth? Justify with specific reason.

Producers should add their movies in the month when audience is more responsive.
Although no specific trend is shown but most movies should be uploaded in year end with some discount in the subscription.
## Chart - 12(What is the Day-wise number of content added in each year on Netflix?)

```
plt.figure(figsize=(20,8))
df_year_month = df.groupby(['year_added','day_added']).agg({'title':'nunique'}).reset_index().sort_values(by=['year_added'],ascending=False)
sns.lineplot(x = 'year_added',y='title', data = df_year_month, palette = 'hls', hue=df_year_month['day_added'], marker='o')
plt.grid(linestyle='--', linewidth=0.3)
plt.show()
```

## 1. Why did you pick the specific chart?
This graph helps us in knowing which day is more frequent in movie addition year wise.

## 2. What is/are the insight(s) found from the chart?
Movies from 2008 to 2009 added on 5th day of the month.
Movies from 2009 to 2010 added on 15th day of the month.
Most of the movies from 2010 to 20012 added in month end.
From 2015 onwards most of the movies are added in month end or mid month.
## 3. Will the gained insights help creating a positive busniness impact?
Are there any insights that lead to negative growth? Justify with specific reason.

Currently most of the movies are added in 15th day of month or at the last day of month, so before releasing the movies consider this trend also.

## Chart - 13(What is Distribution of Duration of contents over Netflix?)

```
#Checking the distribution of Movie Durations
plt.figure(figsize=(10,7))
plots= sns.distplot(df_movies['duration'],kde=False, color=['green'])
plt.title('Distplot with Normal distribution for Movies',fontweight="bold")
for bar in plots.patches:
   plots.annotate(bar.get_height(),
                  (bar.get_x() + bar.get_width() / 2,
                   bar.get_height()), ha='center', va='bottom',
                  size=7, xytext=(0, 5),
                  textcoords='offset points', rotation=90)
plt.show()

plt.figure(figsize=(23,8))
df_duration = df_tvshows.groupby(['duration']).agg({'title':'nunique'}).reset_index().sort_values(by=['duration'],ascending=False)
plots= sns.barplot(x = 'duration',y='title', data = df_duration, palette='husl')
plt.title(f'Barplot of TV Shows Duration')
plt.ylabel(f"Content count")
plt.grid(linestyle='--', linewidth=0.3)
for bar in plots.patches:
   plots.annotate(bar.get_height(),
                  (bar.get_x() + bar.get_width() / 2,
                   bar.get_height()), ha='center', va='bottom',
                  size=12, xytext=(0, 8),
                  textcoords='offset points', rotation=90)
plt.show()
```

## 1. Why did you pick the specific chart?
To know the duration distribution for Movies and TV Shows on Netflix.

## 2. What is/are the insight(s) found from the chart?
The histogram of the distribution of movie durations in minutes on Netflix shows that the majority of movies on Netflix have a duration beyween 80 to 120 minutes.

The countplot of the distribution of TV Show durations in seasons on Netflix shows that the most common duration for TV show on Netflix is one season, followed by two seasons.

## Chart - 14 (What is the Distribution of Content Rating in each highest content creating countries?)

```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df['count'] = 1
data = df.groupby('Country')[['Country', 'count']].sum().sort_values(by='count', ascending=False).reset_index()[:10]
data = data['Country']
df_heatmap = df.loc[df['Country'].isin(data)]
df_heatmap = pd.crosstab(df_heatmap['Country'], df_heatmap['rating'], normalize="index").T

# Plotting the heatmap
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Defining order of representation
country_order = ['United States', 'India', 'United Kingdom', 'Canada', 'France', 'Japan', 'Spain', 'South Korea', 'Germany', 'Mexico']
rating_order = ['Adult Content', 'Teen Content', 'Children Content', 'Family-friendly Content', 'Not Rated']

# Calling and plotting heatmap
sns.heatmap(df_heatmap.loc[rating_order, country_order], cmap="jet", square=True, linewidth=2.5, cbar=False, annot=True, fmt='1.0%',
            vmax=.6, vmin=0.05, ax=ax, annot_kws={"fontsize": 12})
plt.show()
```

## 1. Why did you pick the specific chart?
This graph shows us which countries producing which type of content the most.

## 2. What is/are the insight(s) found from the chart?
We found that most of the countries produces content related to Adult and Teen.
Among all the countries India has less content in Adult segment than teen content.
85% of content is Adult content from Spain.
Canada produces more content related to Children and Family-friendly content.
## 3. Will the gained insights help creating a positive busniness impact?
Are there any insights that lead to negative growth? Justify with specific reason.

Companies should target the country audience according to their taste of content choice.
As in Spain Production house should more focus on Adult Content.
Production house should more focus on children and Family-friendly content for Canada because there are chances of growth.
5. Hypothesis Testing
Based on your chart experiments, define three hypothetical statements from the dataset. In the next three queestions, perform hypothesis testing to obtain final conclusion about the statements through your code and statistical testing.

# Hypothetical Statement 1:

Null Hypothesis: There is no significant difference in the proportion ratings of drama movies and comedy movies on Netflix.
Alternate Hypothesis: There is significant difference in the proportion ratings of drama movies and comedy movies on Netflix.
Hypothetical Statement 2:

* Null Hypothesis: The average duration off TV shows added in the year 2020 on Netflix is not significantly different from the average duration of TV shows added in the year 2021.
* Alternative Hypothesis: The average duration off TV shows added in the year 2020 on Netflix is significantly different from the average duration of TV shows added in the year 2021.
Hypothetical Statement 3:

* Null Hypothesis: The proportion of TV shows added on Netflix that are produced on United States is not significantly different from the proportion of movies added on Netflix that are produced in the United States.
* Alternative Hypothesis: The proportion of TV shows added on Netflix that are produced on United States is significantly different from the proportion of movies added on Netflix that are produced in the United States.
# Hypothetical Statement - 1
## 1. State Your research as a null hypothesis and alternate hypothesis.
* Null Hypothesis: There is no significant difference in the proportion ratings of drama movies and comedy movies on Netflix.
* Alternative Hypothesis: There is a significant difference in the proportion ratings of drama movies and comedy movies on Netflix.
# 2. Perform an appropriate statistical test.

```
# Perform Statistical Test to obtain P-Value
from statsmodels.stats.proportion import proportions_ztest  #------> This function is used to perform z test of proportion.

# Subset the data to only include drama and comedy movies
subset = df[df['Genre'].str.contains('Dramas') | df['Genre'].str.contains('Comedies')]

# Calculate the proportion of drama and comedy movies
drama_prop = len(subset[subset['Genre'].str.contains('Dramas')]) / len(subset)
comedy_prop = len(subset[subset['Genre'].str.contai
```
z-statistic:  64.8000705213286
p-value:  0.0
Reject the null hypothesis.

We conclude that there is a significant difference in the proportion ratings of drama movies and comedy movies on Netflix.

## Which statistical test have you done to obtain P-Value?
The statistical test we have used to obtain the P-value is the z-test for proportions.

## Why did you choose the specific statistical test?
The z-test for proportions was choosen because we are comparing the proportions of two categorical variables (drama movies and comedy movies) in a single. The null hypothesis and alternative hypothesis are about the difference in proportions and we want to determine if the observed difference in proportions is stastically significant or not. The z-test for proportions is appropriate for this situation because it allows us to compare two proportions and calculate the probability and calculate the probability of observing the difference we see in our sample if the null hypothesis were true.:

# Hypothesis Statement - 2
## 1. State Your research hypothesis as a null hypothesis and alternate hypothesis.
* Null Hypothesis: The average duration of TV Shows added in the year 2020 on Netflix is not significantly different from the average duration of TV shows added in the year 2021.

* Alternative Hypothesis: The average duration of TV Shows added in the year 2020 on Netflix is significantly different from the average duration of TV shows added in the year 2021.

# 2. Perform an appropriate statistical test.

```
# Perform Statistical Test to obtain P-Value
# To test this hypothesis, we perform a two-sample t-test.
from scipy.stats import ttest_ind

# Create seperate dataframes for TV shows in 2020 and 2021
tv_2020 = df[(df['type'] == 'TV Show') & (df['release_year'] == 2020)]
tv_2021 = df[(df['type'] == 'TV Show') & (df['release_year'] == 2021)]

# Perform two-sample t-test
t, p = ttest_ind(tv_2020['duration'].astype(int),
```

t_value:  -6.002151232542292
p_value:  7.23381843379902e-09
Reject null hypothesis. 

The average duration of TV shows added in the year 2020 on Netflix is significantly different from the average duration of TV shows added in the year 2021.
## Which statistical test have you done to obtain P-Value?
The statistical test used to obtain the P-Value is a two sample t-test.

## Why did you choose the specific statistical test?
The two-sample t-test was choosen because we are comparing the means of two different samples (TV shows added in 2020 as TV shows added in 2021) to determine whether they are significantly different. Additionally, we assume that the two samples have unequal variances since it is unlikely that the duration of TV shows added in 2020 and 2021 would have the exact same variance.

# Hypothetical Statement - 3
## 1. State Your research hypothesis as a null hypothesis and alternate hypothesis.
* Null Hypothesis: The proportion of TV Shows added on Netflix that are produced in the United States is not significantly different from the proportion of movies added on Netflix that are produced in the United States.

* Alternative Hypothesis: The proportion of TV Shows added on Netflix that are produced in the United States is significantly different from the proportion of movies added on Netflix that are produced in the United States.

## 2. Perform an appropriate statistical test.

```
# Perform Statistical Test to obtain P-Value
from statsmodels.stats.proportion import proportions_ztest  #------> This function is used to perform z test of proportion.

# Calculate the proportion of drama and comedy movies
tv_proportion = np.sum(df_tvshows['Country'].str.contains('United States')) / len(df_tvshows)
movie_proportion = np.sum(df_movies['Country'].str.contains('United States')) / len(df_movies)

# Set up the parameters for the z-test
count = [int(tv_proportion * len(df_tvshows)), int(movie_proportion * len(df_movies))]
nobs = [len(df_tvshows), len(df_movies)]
alternative = 'two-sided'

# Perform the z-test
z_stat, p_value = proportions_ztest(count=count, nobs=nobs, alternative=alternative)
print('z-statistic: ', z_stat)
print('p-value: ', p_value)

# Set the significance level
alpha = 0.05

# Print the results of the z-test
if p_value < alpha:
    print(f"Reject the null hypothesis.")
else:
    print(f"Fail to reject the null hypothesis.")
```

z-statistic:  -4.838078469799881
p-value:  1.3110038583414833e-06
Reject the null hypothesis.
We conclude that the proportionof TV shows added on Netflix that are produced in the United States is significantly different from the proportion of movies added on Netflix that are produced in the United States.

## Which statistical test have you done to obtain P-Value?
The statistical test used to obtain P-Value is a two sample proportion test.

## Why did you choose the specific statistical test?
We chose this specific statistical test because is is appropriate for comparing two proportions and it helps us to determine whether the difference between the two proportions is due to chance or not.

# 6. Feature Engineering & Data Pre-Processing
Handling Missing Values

```
# Handling Missing Values * Missing Value Imputation
# Since we have already dealed with null value. So it is not neede now.
df.isna().sum()
```

title           0
Actors          0
Directors       0
Genre           0
Country         0
type            0
release_year    0
rating          0
duration        0
description     0
day_added       0
month_added     0
year_added      0
count           0
dtype: int64
Let's move ahead, as we have already dealed with null/missing values from our dataset.

# 2. handling Outliers

```
# Storing the continous values feature in a seperate list
continous_value_feature= ["release_year", "duration", "day_added", "month_added", "year_added"]

# Checking outliers with the help of box plot for continous features
plt.figure(figsize=(16,5))
for n,column in enumerate(continous_value_feature):
  plt.subplot(1, 5, n+1)
  sns.boxplot(df[column])
  plt.title(f'{column.title()}',weight='bold')
  plt.tight_layout()
```

Although we have some of the anomalies in continous feature but we will not treat by considering outliers as some of the Movies/TV Shows has released or added early on Netflix

# 3. Textual Data Preprocessing

```
# Here we are taking the copied dataframe as the data having more number of observations resulted in ram exhaustion.
df.shape, df_new.shape
```

((175807, 14), (7770, 12))

```
# Binning of rating in new dataframe
df_new['rating'].replace(rating_map, inplace = True)

# Checking sample after binning
df_new.sample(2)
```
# 1. Textual Columns
```
# Craeting new feature content_detail with the help of other textual attributes
df_new["content_detail"]= df_new["cast"]+" "+df_new["listed_in"]+" "+df_new["rating"]+" "+df_new["country"]+" "+df_new["description"]

# Checking the manipulation
df_new.head(5)
```

# 2. Lower casing
```
# Lower Casing
df_new['content_detail']= df_new['content_detail'].str.lower()

# Checking the manipulation
df_new.iloc[281,]['content_detail']
```

# 3. Removing Puntuation

```
# function to remove punctuations
def remove_punctuations(text):
    '''This function is used to remove the punctuations from the given sentence'''
    #imorting needed library
    import string
    # replacing the punctuations with no space, which in effect deletes the punctuation marks.
    translator = str.maketrans('', '', string.punctuation)
    # return the text stripped off punctuation marks
    return text.translate(translator)

# Removing Punctuations from the content_detail
df_new['content_detail']= df_new['content_detail'].apply(remove_punctuations)

# Checking the observation after manipulation
df_new.iloc[281,]['content_detail']
```

# 4. Removing URLs & Removing words and digits contain digits.

```
def remove_url_and_numbers(text):
    '''This function is used to remove the URL's and Numbers from the given sentence'''
    # importing needed libraries
    import re
    import string

    # Replacing the URL's with no space
    url_number_pattern = re.compile(r'https?://\S+|www\.\S+')
    text= re.sub(url_number_pattern,'', text)

    # Replacing the digits with one space
    text = re.sub('[^a-zA-Z]', ' ', text)

    # return the text stripped off URL's and Numbers
    return text

# Remove URLs & Remove words and digits contain digits
df_new['content_detail']= df_new['content_detail'].apply(remove_url_and_numbers)

# Checking the observation after manipulation
df_new.iloc[281,]['content_detail']
```

# 5. Removing Stopwords & Removing White spaces

```
# Downloading stopwords
nltk.download('stopwords')

# create a set of English stop words
stop_words = stopwords.words('english')

# displaying stopwords
print(stop_words)
```

['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data]   Unzipping corpora/stopwords.zip.

```
def remove_stopwords_and_whitespaces(text):
    '''This function is used for removing the stopwords from the given sentence'''
    text = [word for word in text.split() if not word in stopwords.words('english')]

    # joining the list of words with space separator
    text=  " ".join(text)

    # removing whitespace
    text = re.sub(r'\s+', ' ', text)

    # return the manipulated string
    return text

# Remove URLs & Remove words and digits contain digits
df_new['content_detail']= df_new['content_detail'].apply(remove_stopwords_and_whitespaces)

# Checking the observation after manipulation
df_new.iloc[281,]['content_detail']

df_new['content_detail'][0]
```

# 6. Tokenization

```
# Downloading needed libraries
nltk.download('punkt')

# Tokenization
df_new['content_detail']= df_new['content_detail'].apply(nltk.word_tokenize)

# Checking the observation after manipulation
df_new.iloc[281,]['content_detail']
```

[nltk_data] Downloading package punkt to /root/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
['tamannaah',
 'bhatia',
 'prabhu',
 'deva',
 'sonu',
 'sood',
 'sapthagiri',
 'murli',
 'sharma',
 'rv',
 'udhaykumar',
 'joy',
 'mathew',
 'hema',
 'comedies',
 'international',
 'movies',
 'scifi',
 'fantasy',
 'teen',
 'content',
 'india',
 'due',
 'family',
 'pressure',
 'corporate',
 'man',
 'reluctantly',
 'marries',
 'woman',
 'village',
 'new',
 'home',
 'abruptly',
 'assumes',
 'different',
 'persona']
# 7. Text Normalization

```
# Normalizing Text (i.e., Stemming, Lemmatization etc.)
# Importing WordNetLemmatizer from nltk module
from nltk.stem import WordNetLemmatizer

# Creating instance for wordnet
wordnet  = WordNetLemmatizer()

def lemmatizing_sentence(text):
    '''This function is used for lemmatizing (changing the given word into meaningfull word) the words from the given sentence'''
    text = [wordnet.lemmatize(word) for word in text]

    # joining the list of words with space separator
    text=  " ".join(text)

    # return the manipulated string
    return text
[
# Downloading needed libraries
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Rephrasing text by applying defined lemmatizing function
df_new['content_detail']= df_new['content_detail'].apply(lemmatizing_sentence)

# Checking the observation after manipulation
df_new.iloc[281,]['content_detail']
```

# Which text normalization technique have you used and why?
We have used Lemmatization insted of stemming for our project because:

Lemmatization produces a more accurate base word: Unlike stemming, which simply removes the suffix from a word, Lemmatization looks at the meaning of the word and its context to produce a more accurate base form.

Lemmatization can handle different infections: Lemmatization can handle various infections of a word, including plural forms, verb tenses and comparitive forms, making it useful for natural language processing.

Lemmatization produces real words: Lemmatization always produces a real word that can be found in a dictionary, making it easier to interpret the results of text analysis.

Lemmatization improves text understanding: By reducing words to their base form, Lemmatization makes it easier to understand the context and meaning of a sentence.

Lemmatization supports multiple language: While Stemming may only work well for English, Lemmatization is effective for many different languages, making it a more verstile text processing technique.

# 8. Part of speech tagging

```
# tokenize the text into words before POS Taging
df_new['pos_tags'] = df_new['content_detail'].apply(nltk.word_tokenize).apply(nltk.pos_tag)

# Checking the observation after manipulation
df_new.head(5)
```

# 9. Tet Vectorization

```
# Vectorizing Text
# Importing needed libraries
from sklearn.feature_extraction.text import TfidfVectorizer

# Creating instance
tfidfv = TfidfVectorizer(max_features=30000) 
[140]
1s
# Fitting on TfidfVectorizer
x= tfidfv.fit_transform(df_new['content_detail'])

# Checking shape of the formed document matrix
print(x.shape)
```

(7770, 30000)
Which text vectorization technique have you used and why?
We have used TFIDF vectorization in place of BAG OF WORDS because Tf-idf vectorization takes into account the importance of each word in a document. TF-IDF also assigns higher values to rare words that are unique to a particular document, making them more important in the representation.

# 4. Dimensionality Reduction
Do you think that dimensionality reduction is needed? Explain why?
In textual data processing, there are 30,000 attruibutes are created in text vectorization and this huge amount of columns cannot be added with our local machines. So, we will using the Principal Component Analysis (PCA) techniques to reduce the dimensions of this huge sparse matrix.

```
# Dimensionality Reduction
# Importing PCA from sklearn
from sklearn.decomposition import PCA

# Defining PCA object with desired number of components
pca = PCA()

# Fitting the PCA model
pca.fit(x.toarray())

# percent of variance captured by each component
variance = pca.explained_variance_ratio_
print(f"Explained variance: {variance}")
```

Explained variance: [7.73929901e-03 4.54125074e-03 3.66764284e-03 ... 4.39520262e-35
 2.33165321e-35 1.20022085e-39]

```
# Ploting the percent of variance captured versus the number of components in order to determine the reduced dimensions
fig, ax = plt.subplots()
ax.plot(range(1, len(variance)+1), np.cumsum(pca.explained_variance_ratio_))
ax.set_xlabel('Number of Components')
ax.set_ylabel('Percent of Variance Captured')
ax.set_title('PCA Analysis')
plt.grid(linestyle='--', linewidth=0.3)
plt.show()
```

It is clear that the above plot that 7770 principal components can capture the 100% of variance. For our case we will consider only those number of PC's that can capture 95% of variance.

```
## Now we are passing the argument so that we can capture 95% of variance.
# Defining instance
pca_tuned = PCA(n_components=0.95)

# Fitting and transforming the model
pca_tuned.fit(x.toarray())
x_transformed = pca_tuned.transform(x.toarray())

# Checking the shape of transformed matrix
x_transformed.shape
```

(7770, 5993)
## Which dimensionality reduction technique have you used and why?
We have used PCA (Principal Component Analysis) for dimensionality reduction. PCA is widely used technique for reducing the dimensionality of high-dimensional data sets while retaining most of the information in the original data.

PCA works by finding the principal components of the data, which are linear combinations of the original features that capture the maximum amount of variation in the data. By projecting the data onto these principal components, PCA can reduce the number of dimensions while retaining most of the information in the original data.

PCA is a popular choice for dimensionality reduction because it is simple to implemnt, computationally efficient and widely available in most data analysis software packages. Additionally, PCA has been extensively studied and has a strong theoretical foundation, making it a reliable and wwell-understood method.

# 7. ML Model Implementation
## ML Model - 1 (K-Means Clustering)
K-means clustering is a type of unsupervised machine learning algorithm used for partitioning a dataset into K clusters based on similarity of data points. The goal of the algorithm is to minimize the sum of squared distances between each data point and its corresponding cluster centroid. it works iteratively by assigning each data point to its nearestcentroid and then re-computing the centroid of each cluster based on the new assignments. The algorithm terminates when the cluster assignments no longer change or when a maximum of iterations is reached

Let's just iterate over a loop of 1 to 16 clusters and try to find the optimal number of clusters with ELBOW method.

```
## Determining optimal value of K using KElbowVisualizer
# Importing needed library
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

# Instantiate the clustering model and visualizer
model = KMeans(random_state=0)
visualizer = KElbowVisualizer(model, k=(1,16),locate_elbow=False)

# Fit the data to the visualizer
visualizer.fit(x_transformed)

# Finalize and render the figure
visualizer.show()
```

Here it seems that the elbow is forming at the 2 clusters but before blindly believing it let's plot one more chart that iterates over the same number of clusters and determines the Silhouette Score at every point.

Okay, but what is Silhouette Score?

The Silhouette Score is a measure of how similar an object is to its own cluster compared to other clusters. It is used to evaluate the quality of clustering, where a higher score indicates that object are more similar to their own cluster and dissimilar to other clusters.

The Silhouette Score ranges from -1 to 1, where a score of 1 indicates that the object is well-matched to its own cluster and poorly-matched to neighbouring clusters. Conversley, a core -1 indicates that the object is poorly-matched to its own cluster and weel-matched to neighbouring clusters.

```
## Determining optimal value of K using KElbowVisualizer
# Importing needed library
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

# Instantiate the clustering model and visualizer
visualizer = KElbowVisualizer(model, k=(2,16), metric='silhouette', timings=True, locate_elbow=False)

# Fit the data to the visualizer
visualizer.fit(x_transformed)

# Finalize and render the figure
visualizer.show()
```

```
## Computing Silhouette score for each k
# Importing needed libraries
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Defining Range
k_range = range(2, 7)
for k in k_range:
    Kmodel = KMeans(n_clusters=k)
    labels = Kmodel.fit_predict(x_transformed)
    score = silhouette_score(x, labels)
    print("k=%d, Silhouette score=%f" % (k, score))
```

k=2, Silhouette score=0.004376
k=3, Silhouette score=0.005369
k=4, Silhouette score=0.006069
k=5, Silhouette score=0.006802
k=6, Silhouette score=0.004979
From the above plots (Elbow plot and Silhouette plot) it is very clear that the silhouette score is comparitevely good for 4 numbers of clusters, so we will consider in kmeans analysis.

Now let's plot and see how our data points look like after assigning to their respective clusters.

```
#training the K-means model on a dataset
kmeans = KMeans(n_clusters=4, init='k-means++', random_state= 0)

#predict the labels of clusters.
plt.figure(figsize=(10,6), dpi=120)
label = kmeans.fit_predict(x_transformed)
#Getting unique labels
unique_labels = np.unique(label)

#plotting the results:
for i in unique_labels:
    plt.scatter(x_transformed[label == i , 0] , x_transformed[label == i , 1] , label = i)
plt.legend()
plt.show()
```

We have 4 different clusters but unfortunately the above plot is in TWO-DIMENSIONAL. Let's plot the above figure in 3Dusing mplot3d library and see if we are getting the seperated clusters.

```
# Importing library to visualize clusters in 3D
from mpl_toolkits.mplot3d import Axes3D

# Plot the clusters in 3D
fig = plt.figure(figsize=(20,8))
ax = fig.add_subplot(111, projection='3d')
colors = ['r', 'g', 'b', 'y']
for i in range(len(colors)):
    ax.scatter(x_transformed[kmeans.labels_ == i, 2], x_transformed[kmeans.labels_ == i, 0], x_transformed[kmeans.labels_ == i, 1], c=colors[i])

```

## Cool, we can easily differentiate all the 4 clusters with naked eye. Now let's assign the 'Content' in their respective cluster by appending 1 more attribute in the final dataframe.

```
# Add cluster values to the dateframe.
df_new['kmeans_cluster'] = kmeans.labels_
```

## 1. Explain the ML Model used and it's performance?
Starting with defining a function that plot a wordcloud for each of the attribute in the given dataframe.

```
def kmeans_wordcloud(cluster_number, column_name):
    '''function for Building a wordcloud for the movie/shows'''

    #Importing libraries
    from wordcloud import WordCloud, STOPWORDS

    # Filter the data by the specified cluster number and column name
    df_wordcloud = df_new[['kmeans_cluster', column_name]].dropna()
    df_wordcloud = df_wordcloud[df_wordcloud['kmeans_cluster'] == cluster_number]
    df_wordcloud = df_wordcloud[df_wordcloud[column_name].str.len() > 0]

    # Combine all text documents into a single string
    text = " ".join(word for word in df_wordcloud[column_name])

    # Create the word cloud
    wordcloud = WordCloud(stopwords=set(STOPWORDS), background_color="black").generate(text)

    # Convert the wordcloud to a numpy array
    image_array = wordcloud.to_array()

    # Return the numpy array
    return image_array

# Implementing the above defined function and plotting the wordcloud of each attribute
fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(20, 15))
for i in range(4):
    for j, col in enumerate(['description', 'listed_in', 'country', 'title']):
        axs[j][i].imshow(kmeans_wordcloud(i, col))
        axs[j][i].axis('off')
        axs[j][i].set_title(f'Cluster {i}, {col}',fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

# ML Model - 2(Hierarchial Clustering)
Hierarchial Clustering is a type of clustering algorithm used for grouping similar data points together into clusters based on their similarity, by recursively merging or dividing clusters based on a measure of similarity or distance between them.

Let's dive into it by plotting a Dendogram and then we will determine the optimal number of clusters.

```
#importing needed libraries
from scipy.cluster.hierarchy import linkage, dendrogram

# HIERARCHICAL CLUSTERING
distances_linkage = linkage(x_transformed, method = 'ward', metric = 'euclidean')
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('All films/TV shows')
plt.ylabel('Euclidean Distance')
```


## Cool, but what is Dendrogram and how to determine the optimal value of clusters?

A dendrogram is a tree-like diagram that records the sequence of merges or splits. More the distance of the vertical lines in the dendrogram, more the distance between those clusters.
From the above Dendrogram we can say that optimal value of clusters is 2. But before assigning the values to respective clusters, let's check the silhouette scores using Agglomerative clustering and follow the bottom up approach to aggregate the datapounts.

```
## Computing Silhouette score for each k
# Importing needed libraries
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Range selected from dendrogram above
k_range = range(2, 10)
for k in k_range:
    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(x_transformed)
    score = silhouette_score(x, labels)
    print("k=%d, Silhouette score=%f" % (k, score))
```

k=2, Silhouette score=0.001238
k=3, Silhouette score=0.001787
k=4, Silhouette score=0.002057
k=5, Silhouette score=0.000211
k=6, Silhouette score=0.000631
k=7, Silhouette score=0.001106
k=8, Silhouette score=0.001704
k=9, Silhouette score=0.002197
From the above silhouette scores it is clear that the 2 clusters are optimal value (maximum silhouette score), which is also clear from the above Dendrogram that the 2 clusters the eiclidean distances are maximum.

Let's again plot the chart observe the 2 different formed clusters.

```
#training the K-means model on a dataset
Agmodel = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')

#predict the labels of clusters.
plt.figure(figsize=(10,6), dpi=120)
label = Agmodel.fit_predict(x_transformed)
#Getting unique labels
unique_labels = np.unique(label)

#plotting the results:
for i in unique_labels:
    plt.scatter(x_transformed[label == i , 0] , x_transformed[label == i , 1] , label = i)
plt.legend()
plt.show()
```
Again plotting the 3 Dimensional plot to see the clusters clearly.
```
# Importing library to visualize clusters in 3D
from mpl_toolkits.mplot3d import Axes3D

# Plot the clusters in 3D
fig = plt.figure(figsize=(20,8))
ax = fig.add_subplot(111, projection='3d')
colors = ['r', 'g', 'b', 'y']
for i in range(len(colors)):
    ax.scatter(x_transformed[Agmodel.labels_ == i, 0], x_transformed[Agmodel.labels_ == i, 1], x_transformed[Agmodel.labels_ == i, 2],c=colors[i])
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
plt.show()
```

**Cool, we can again easily differentiate the all 2 clusters with naked eye. Now let's assign the 'Content(Movies and TV Shows)' in their respective cluster by appending 1 more attribute in the final dataframe.**

```
# Add cluster values to the dateframe.
df_new['agglomerative_cluster'] = Agmodel.labels_
```

## 1. Explain the ML Model used and it's performance using Evaluation metric Score Chart.
Let's just again define a function that plots wordcloud for different attributes using Agglomerative Clustering.

```
def agglomerative_wordcloud(cluster_number, column_name):
  '''function for Building a wordcloud for the movie/shows'''

  #Importing libraries
  from wordcloud import WordCloud, STOPWORDS

  # Filter the data by the specified cluster number and column name
  df_wordcloud = df_new[['agglomerative_cluster', column_name]].dropna()
  df_wordcloud = df_wordcloud[df_wordcloud['agglomerative_cluster'] == cluster_number]

  # Combine all text documents into a single string
  text = " ".join(word for word in df_wordcloud[column_name])

  # Create the word cloud
  wordcloud = WordCloud(stopwords=set(STOPWORDS), background_color="black").generate(text)

  # Return the word cloud object
  return wordcloud

# Implementing the above defined function and plotting the wordcloud of each attribute
fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(20, 15))
for i in range(2):
    for j, col in enumerate(['description', 'listed_in', 'country', 'title']):
        axs[j][i].imshow(agglomerative_wordcloud(i, col))
        axs[j][i].axis('off')
        axs[j][i].set_title(f'Cluster {i}, {col}',fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

# ML model - 3(Building a Recommendation System)
We are using Cosinr similarity as it is a measure of similarity between two non-zero vectors in a multidimensional spaces. It measures the cosine of the angle between the two vectors, which ranges from -1 (opposite direction) to 1 (same direction) with 0 indicating orthogonality (the vecrtors are perpendicular to each other).

In this project we have used cosine similarity which is used to determine how similar two documents or pieces of text are. We represent the documents as vectors in a high-dimensional space, where each dimension represents a word or term in the corpus. We can then calculate the cosine similarity between the vectors to determine how similar the documents are based on their word usage.

We are using cosine similarity over tf-idf because:

Cosine similarity handles high dimensional sparse data better.
Cosine similarity captures the meaning of the text better than the tf-idf. For example, if two items similar words but in different orders, cosine similarity would still consider them similar, while tf-idf may not. This is because tf-idf only considers the frequency of words in a document and not their order or meaning.

```
# Importing neede libraries
from sklearn.metrics.pairwise import cosine_similarity

# Create a TF-IDF vectorizer object and transform the text data
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_new['content_detail'])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix)
```

Let's check how our recommender sustem is performing.

```
# Testing indian movie
recommend_content('Kal Ho Naa Ho')


# Testing non indian movie
recommend_content('Zombieland')


# Testing indian tv show
recommend_content('Zindagi Gulzar Hai')

# Testing non indian tv show
recommend_content('Vampires')
```

## 1. Which Evaluation metrics did you consider for a positive business impact and why?
We have choosen Silhoutte Score and Distortion Score (also known as inertia or sum od squared distances) as evaluation metrics as it measures how well each data point in a clustr is separted from other clusters. It ranges from -1 to 1, with higher values indicating better cluster separation. A silhouette score close to 1 indicates that the data point is well-matched to its own cluster and poorly matched to neifgbouring cluster. A score close to 0 indicates taht the data point probably assigned to the wrong cluster.

The advantages of using silhouette score over distortion score are:

Silhouette score takes into account both the cohesion (how well data points within a cluster are similar) and separation (how well data points in different clusters are dissimilar) of the clusters, whereas distortion score only considers the compactness of each other.
Silhouette score is less sensitive to the shape of the clusters, while distortion score tends to favor spherical clusters, and in our case the clusters are not completely spherical.
Silhouette score provides more intutive and interpretable results, as it assigns a score to each data point rather than just a single value for the entire clustering solution.
2. Which ML Model did you choose from the above created models as your final predection model and why?
We have consider K-means as our final model, as we are getting the comparitively high Silhouette Score in K-means clustering and the resulted clusters are very well seperated from each others as have seen in the 3 dimensions.

Also in some of the siyuations K-means works accurately than other clustering methods such as:

Speed: K-means is generally faster than hierarchical clustering, especially when dealing with large datasets, since it involves fewer calculations and iterations.

Ease of use: K-means is relatively starightforward to implement and interpret as it requires only few parameters (such as the number of clusters) and produces a clear partioning of the data.

Scalibility: K-means can easily handle daatsets with a large number of variables or dimensions, where as hierarchical clustering becomes computationally expensive as the number of data points and dimensions increases.

Independence of clusters: K-means produces non-overlapping clusters, whereas hierarchical clustering can produce overlapping clusters or clusters that are nested within each other, which may not be ideal for certain applications.

# Conclusion
Conclusion drawn from EDA
Based on the exploratory data analysis (EDA) of the Netflix movie and TV Shows clustering dataset, we have drawn from the following conclusions:

Movies make up about two-thirds of Netflix content, with TV Shows comprising the remaining one third.
Adult and tee categories are prevalent on Netflix, while family-friendly content is more common in TV shows than in movies.
Indian actorsdominate Netflix movies, while popular indian actors are absent from TV shows.
Jam Suter is the most common movie directoe, and Ken Burns is the most common TV show dorector on Netflix.
The United States is the largest producer of movies and TV shows on Netflix, followed by India, Japan and South Korea have more TV shows than movies, indicating growth potential in that area.
International movies, drama, and comedy are the msot popular genres on Netflix.
Tv shows additions on Netflix have increased since 2018, while movie additions have decreased. In 2020, fewer movies were added compared to 2019, but more TV shows were added.
October, November and December are popular months for adding Tv shows, while January, October and November are popular for adding movies. February sees the least additions.
Movies and TV shows are typically added at the beginning or middle of the month and are popularly added on weekends.
Most movies on Netflix have duration between 80 to 120 minutes, while TV shows commonly have one or two seasons.
Various countries contribute adult and teen content, with Spain producing the most adult content and Canada focusing on Children and family-friendly categories.
Conclusins drawn from ML Model
Implemented K-Means Clustering and Agglomerative Hierarchical Clustering, to cluster the Netflix movies TV show dataset.
The optimal number of clusters wwe are getting fron K-Means is 4, whereas for Agglomerative Hierarchical Clustering the optimal number of clusters are found out to be 2.
We chose Silhouette Score as the evaluation metric over distortion score because it provides a more intitive and interpretable result. Also Silhouette score is less sensitive to the shape of the clusters.
Built a Recommendation system that can help Netflix improves user experience and reduce subscriber churn by providing personalized recommendations to users based to their similarity scores.
Future Work (Optional)
Integrating this dataset with external sources such as IMDB ratings, books clustering, Plant based Type clustering can lead to numerous intriguing discoveries.
By incorporating additional data, a more comprehensive recommender system could be developed, offering enhances recommendations to users. This system could then be deployed on the web for widespread usage.

## Hurrah! We have successfully completed our Machine Learning Capstone Project !!!
