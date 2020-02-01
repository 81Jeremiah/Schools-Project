#!/usr/bin/env python
# coding: utf-8

# # Read in the data

# In[1]:


import pandas as pd
import numpy
import re

data_files = [
    "ap_2010.csv",
    "class_size.csv",
    "demographics.csv",
    "graduation.csv",
    "hs_directory.csv",
    "sat_results.csv"
]

data = {}

for f in data_files:
    d = pd.read_csv("schools/{0}".format(f))
    data[f.replace(".csv", "")] = d


# # Read in the surveys

# In[2]:


all_survey = pd.read_csv("schools/survey_all.txt", delimiter="\t", encoding='windows-1252')
d75_survey = pd.read_csv("schools/survey_d75.txt", delimiter="\t", encoding='windows-1252')
survey = pd.concat([all_survey, d75_survey], axis=0)

survey["DBN"] = survey["dbn"]

survey_fields = [
    "DBN", 
    "rr_s", 
    "rr_t", 
    "rr_p", 
    "N_s", 
    "N_t", 
    "N_p", 
    "saf_p_11", 
    "com_p_11", 
    "eng_p_11", 
    "aca_p_11", 
    "saf_t_11", 
    "com_t_11", 
    "eng_t_11", 
    "aca_t_11", 
    "saf_s_11", 
    "com_s_11", 
    "eng_s_11", 
    "aca_s_11", 
    "saf_tot_11", 
    "com_tot_11", 
    "eng_tot_11", 
    "aca_tot_11",
]
survey = survey.loc[:,survey_fields]
data["survey"] = survey


# # Add DBN columns

# In[3]:


data["hs_directory"]["DBN"] = data["hs_directory"]["dbn"]

def pad_csd(num):
    string_representation = str(num)
    if len(string_representation) > 1:
        return string_representation
    else:
        return "0" + string_representation
    
data["class_size"]["padded_csd"] = data["class_size"]["CSD"].apply(pad_csd)
data["class_size"]["DBN"] = data["class_size"]["padded_csd"] + data["class_size"]["SCHOOL CODE"]


# # Convert columns to numeric

# In[4]:


cols = ['SAT Math Avg. Score', 'SAT Critical Reading Avg. Score', 'SAT Writing Avg. Score']
for c in cols:
    data["sat_results"][c] = pd.to_numeric(data["sat_results"][c], errors="coerce")

data['sat_results']['sat_score'] = data['sat_results'][cols[0]] + data['sat_results'][cols[1]] + data['sat_results'][cols[2]]

def find_lat(loc):
    coords = re.findall("\(.+, .+\)", loc)
    lat = coords[0].split(",")[0].replace("(", "")
    return lat

def find_lon(loc):
    coords = re.findall("\(.+, .+\)", loc)
    lon = coords[0].split(",")[1].replace(")", "").strip()
    return lon

data["hs_directory"]["lat"] = data["hs_directory"]["Location 1"].apply(find_lat)
data["hs_directory"]["lon"] = data["hs_directory"]["Location 1"].apply(find_lon)

data["hs_directory"]["lat"] = pd.to_numeric(data["hs_directory"]["lat"], errors="coerce")
data["hs_directory"]["lon"] = pd.to_numeric(data["hs_directory"]["lon"], errors="coerce")


# # Condense datasets

# In[5]:


class_size = data["class_size"]
class_size = class_size[class_size["GRADE "] == "09-12"]
class_size = class_size[class_size["PROGRAM TYPE"] == "GEN ED"]

class_size = class_size.groupby("DBN").agg(numpy.mean)
class_size.reset_index(inplace=True)
data["class_size"] = class_size

data["demographics"] = data["demographics"][data["demographics"]["schoolyear"] == 20112012]

data["graduation"] = data["graduation"][data["graduation"]["Cohort"] == "2006"]
data["graduation"] = data["graduation"][data["graduation"]["Demographic"] == "Total Cohort"]


# # Convert AP scores to numeric

# In[6]:


cols = ['AP Test Takers ', 'Total Exams Taken', 'Number of Exams with scores 3 4 or 5']

for col in cols:
    data["ap_2010"][col] = pd.to_numeric(data["ap_2010"][col], errors="coerce")


# # Combine the datasets

# In[7]:


combined = data["sat_results"]

combined = combined.merge(data["ap_2010"], on="DBN", how="left")
combined = combined.merge(data["graduation"], on="DBN", how="left")

to_merge = ["class_size", "demographics", "survey", "hs_directory"]

for m in to_merge:
    combined = combined.merge(data[m], on="DBN", how="inner")

combined = combined.fillna(combined.mean())
combined = combined.fillna(0)


# # Add a school district column for mapping

# In[8]:


def get_first_two_chars(dbn):
    return dbn[0:2]

combined["school_dist"] = combined["DBN"].apply(get_first_two_chars)


# # Find correlations

# In[9]:


correlations = combined.corr()
correlations = correlations["sat_score"]
print(correlations)


# # Plotting survey correlations

# In[10]:


# Remove DBN since it's a unique identifier, not a useful numerical value for correlation.
survey_fields.remove("DBN")


# In[11]:


get_ipython().magic('matplotlib inline')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt


# # Find Correlations

# In[12]:



correlations = combined.corr()
correlations = correlations["sat_score"]
print(correlations)


# In[13]:


correlations[survey_fields].plot.bar()


# # Correlation Discoveries

# We can see that there is a strong positive correlations with the following:
# 
# - Number of student respondents
# - Number of parent respondents
# - Safety and Respect score based on teacher responses
# - Safety and Respect score based on student responses
# - Academic expectations score based on student responses
# - Safety and Respect total score
# 
# There is also a negative correlation with:
# - Communication score based on parent responses
# 
# From this we can see that reposnce and safetfy all contribute to higher SAT scores. However commincation has an adverse effect and shows lowers scores with communication. 
# 

# In[14]:


combined.plot.scatter('saf_s_11','sat_score' )


# # Safety and Rescpect correlation Scatter Plot

# The 'Safety and Respect score based on student responses' and Sat Score columns show are strong correlation but with a bunch up between 5 & 7 safety and respect score. 
#  

# In[15]:


dist_mean = combined.groupby('school_dist').agg(np.mean)
dist_mean.reset_index(inplace=True)


# In[16]:


saf_s_means = dist_mean['saf_s_11']
saf_s_means


# In[17]:


saf_t_means = dist_mean['saf_t_11']
saf_t_means


# In[ ]:





# In[20]:


from mpl_toolkits.basemap import Basemap
m = Basemap(
    projection='merc', 
    llcrnrlat=40.496044, 
    urcrnrlat=40.915256, 
    llcrnrlon=-74.255735, 
    urcrnrlon=-73.700272,
    resolution='i'
)
m.drawmapboundary(fill_color='#85A6D9')
m.drawcoastlines(color='#6D5F47', linewidth=.4)
m.drawrivers(color='#6D5F47', linewidth=.4)

longitudes = dist_mean["lon"].tolist()
latitudes = dist_mean["lat"].tolist()
m.scatter(longitudes, latitudes, s=50, zorder=2, latlon=True, c=dist_mean['saf_s_11'], cmap='summer')
plt.show()


# By mapping the safety scores we can see that Upper Manhattan and parts of Queens and the Bronx tend to have higher safety scores, whereas Brooklyn( with the exception of 2 districts) has low safety scores.

# # Race Correlations

# In[22]:


race_cols = ['white_per', 'asian_per', 'black_per', 'hispanic_per' ]
correlations[race_cols].plot.bar()


# There is a strong postive correlation between white and asian races and SAT scores. There is also a medium negative correlation between hispanics and SAT scores.

# In[23]:


combined.plot.scatter('hispanic_per', 'sat_score')


# There is a fairly strong correlation between schools with a high hispanic percentage and lower SAT scores. Ther are still some schools with low hispanic percentages and low SAT scores. I is also clear the schools with over a 40% hispanic attendence rate have a score of 1500 or below.

# In[28]:


over_95_hisp = combined[combined['hispanic_per'] > 95]
over_95_hisp['SCHOOL NAME']


# There are 8 schools with a hispanic percentage over 95%. All of these schools are schools that focus on spanish speaking immigrants.

# In[31]:


hisp_less_than_10 = combined[combined['hispanic_per'] < 10]
hisp_less_10_over_1800 = hisp_less_than_10[hisp_less_than_10['sat_score']
                                           > 1800]
hisp_less_10_over_1800['SCHOOL NAME']


# When comparing schools that have less than 10 % hispanic students and SAT scores over 1800 it is visible that there are 5 schools that meet this criteria. These 5 schools are all elite schools that require an entrance exam and focus on students who already have the ability to do well on tests. 

# # Gender and SAT Scores

# In[32]:


correlations[['male_per', 'female_per' ]].plot.bar()


# There is a minor negative correlation between male percentage in a school and SAT scores. This implies that more men in a school, the lower the average SAT scores. 

# In[33]:


combined.plot.scatter('female_per', 'sat_score' )


# The correlation between female percentage and SAT scores does appear weak, however there is a very interesting spike in SAT scores with schools between 40 and 70 % females. 

# In[34]:


fem_over_60 = combined[combined['female_per'] > 60]
fem_over_60_SAT_over_1700 = fem_over_60[fem_over_60['sat_score'] > 1700]
fem_over_60_SAT_over_1700['SCHOOL NAME']


# Schools with female percentages over 60 and SAT scores over 1700 are smaller more elite college prep schools. 

# In[36]:


combined['ap_per'] = (combined['AP Test Takers '] / combined['total_enrollment']) * 100
combined['ap_per'].head()


# In[37]:


combined.plot.scatter('ap_per', 'sat_score')


# There is a weak correlation between ap test takers percentage and SAT scores. There are a few odd outlying schools that event show a very high percentage of AP test takers and lower SAT scores. 
