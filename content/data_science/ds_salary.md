+++
title="Data Science Negotiation"
description="Estimating a fair data science salary"
weight=5
portfolio=true
image="/images/negotiation.jpg"
+++

#  Introduction

O'Really conducts yearly surveys on Data Science and Data Engineers positions highlighting the main factors that can influence data professionals' salaries. In that [survey](http://www.oreilly.com/data/free/2016-data-science-salary-survey.csp) a linear regression model incorporating the most relevant career influencing variables was compiled.

**The ipython [notebook](https://github.com/dkarapetyan/negotiation/blob/master/ds_salary.ipynb) of this report is designed to allow data scientists and data engineers to plug in their own data (see User Parameters section) to find out whether their current 2016 salary is aligned with their market value.**

According to the authors of the survey, their model is able to explain roughly 75% of the variance in the data:

_"Our model explains about three-quarters of the variance in the sample salaries (with an R2 of 0.747). Roughly half of the salary variance is due to geography and experience. Given the important factors that can not be captured in the survey—for example, we don’t measure competence or evaluate the quality of respondents’ work output—it’s not surprising that a large amount of variance is left unexplained."_



# Modules Used


```python
import pandas as pd
import locale
from copy import copy
from decimal import Decimal
locale.setlocale( locale.LC_ALL, 'en_US.UTF-8' )
from IPython.display import display
```

# Model Parameters


```python
usgs = pd.read_csv("usgs_state_2016.csv")
#drop garbage columns
cols_interest = [x for x in usgs.columns if x == 'State' or x=='Gross State Product']
usgs = usgs[cols_interest]
usgs.dropna(inplace=True)
#convert numbers represented as strings to float64 dtypes
usgs.loc[:, 'Gross State Product'] = usgs.loc[:, 'Gross State Product'].apply(locale.atof)
```

# User Parameters:


```python
state = 'California'
gender='Male'
years_experience = 3
bargaining_skill_points = 5 #scale of 1 to 5
age = 34
academic_specialty='Math'
have_phd = 1
currently_student=0
industry='Software'
company_size= 100
company_age=16
coding_amount='over 20 hours/week'
meetings='1 to 3 hours/week'
work_week = 40
job_title='Upper Management'#Senior Engineer/Developer'
etl_involvement='Minor'
setting_up_maintaining_data_platforms='NA'
developing_prototype_models='Major'
developing_hardware='Minor'
organizing_guiding_team_projects='Major'
identifying_bus_analytics_problems='Major'
communicating_people_outside_company='Major'
most_work_done_with_cloud_computing=1
languages_used='Python'
tools_used='Unix, PostgreSQL, SQLite, MongoDB, Redshift, Spark, Hive, Spark Mlib'
```

# Weighting:

## Scaler Functions and Dictionaries


```python
def age_scaler(age):
    scale=None
    if 26 <= age <=30:
        scale = 17.2
    elif 31 <= age <= 35:
        scale = 22.5
    elif 36 <= age <= 65:
        scale = 38.5
    return scale

def company_size_scaler(size):
    if x < 0:
        raise ValueError("No negatives allowed")
    elif 0 <= x <= 500:
        return 0
    elif 501 <= x <= 10000:
        return 3.6
    else:
        return 7.7

def coding_amount_scaler(amt):
    if 0 <= x < 1:
        return 0
    elif 1 <= x <= 3:
        return -8.2
    elif 4 <= x <= 20:
        return -3
    else:
        return -0.5

def meeting_scaler(num_meetings):
    if x < 1:
        return 0
    elif 1 < x <= 3:
        return 1
    elif 4 <= x <= 8:
        return 9.2
    elif 9 <= x <= 20:
        return 20.6
    else:
        return 21.1

def work_week_scaler(amt_worked):
    if amt_worked < 46:
        return 0
    elif 46 <= amt_worked <= 60:
        return 1
    else:
        return -2.4



def languages_used_scaler(array_langs):
    sum = 0
    if 'Python' in array_langs:
        sum+=4.6
    elif 'JavaScript' in array_langs:
        sum+=-2.2
    elif 'Excel' in array_langs:
        sum+=-7.4
    return sum

def tools_used_scaler(tools_array):
    cluster_0 = ['MySQL', 'PostgreSQL', 'SQLite', 'Redshift', 'Vertica', 'Redis', 'Ruby']
    cluster_1 = ['Spark', 'Unix', 'Spark MlLib', 'ElasticSearch', 'Scala', 'H2O', 'EMC/Greenplum', 'Mahout']
    cluster_2 = ['Hive', 'Apache Hadoop', 'Cloudera', 'Hortonworks', 'Hbase', 'Pig', 'Impala']
    cluster_3 = ['Tableau', 'Teradata', 'Netezza (IBM)', 'Microstrategy', 'Aster Data (Teradata)', 'Jaspersoft']
    cluster_4 = ['MongoDB', 'Kafka', 'Cassandra', 'Zookeeper', 'Storm', 'JavaScript InfoVis Toolkit', 'Go', 'Couchbase']
    clusters = [cluster_0, cluster_1, cluster_2, cluster_3, cluster_4]
    cluster_ops=[0, 0, 0, 0, 0]

    for item in tools_array:
        for i in range(5):
            if item in clusters[i]:
                cluster_ops[i]+=1
    cluster_ops_maxed = copy(cluster_ops)
    for i, item in enumerate(cluster_ops):
        if i==0 or i==4:
            cluster_ops[i] = min(item, 4)
        elif i==1 or i==2:
            cluster_ops[i] = min(item, 5)
        else:
            cluster_ops[i] = min(item, 3)

    scaling_factors = [1.7, 3.9, 1.5, 2.4, 1.3]
    scaled = sum([a*b for a,b in zip(cluster_ops_maxed, scaling_factors)])
    return scaled

job_title_scaler = {'Upper Management': 20.2, 'Engineer/Developer/Programmer': -0.9, 'Manager': 3.1, 'Researcher': -1,
                   'Architect': 14.3, 'Senior Engineer/Developer': 4.6}

industry_scales = {'Software': 2.2, 'Banking/Finance': 3.0, 'Advertising/Marketing/PR': -2.0, 'Education': -24.5,
                  'Computers/Hardware': -3.9, 'Search/Social Networking': 7.1}
etl_scaler = {'NA': 0, 'Minor': 4.5, 'Major': -1.9}
developing_prototype_models_scaler={'NA': 0, 'Minor': 4.4, 'Major': 12.1}
developing_hardware_scaler = {'NA': 0, 'Minor': 0, 'Major': -1.3}
organizing_guiding_team_projects_scaler = {'NA': 0, 'Minor': 0, 'Major': 9.7}
ibap_scaler = {'NA': 0, 'Minor': 1.5, 'Major': 6.7}
cpoc_scaler = {'NA': 0, 'Minor': 0, 'Major': 5.4}
data_platforms_scaler = {'NA': 0, 'Minor': -4.9, 'Major': -4.9}
```


```python
results=dict(constant=60,
state_gdp_scaled = 2.6 * usgs[usgs['State'] == state]['Gross State Product'] / float(1000),
gender_scaled = (-7.8 if gender is 'Female' else 0),
years_experience_scaled = 3.8 * years_experience,
bargaining_skill_points_scaled = 7.4 * bargaining_skill_points,
age_scaled = age_scaler(age),
academic_specialty_scaled = 3.9 * (1 if academic_specialty in ['Math', 'Physics', 'Statistics'] else 0),
have_phd_scaled = 12.2 * have_phd,
currently_student_scaled= -9.7 * currently_student,
industry_scaled = industry_scales.get(industry),
company_size_scaled = company_size_scaler(company_size),
company_age_scaled = (-4.3 if company_age > 10 else 0),
coding_amount_scaled = coding_amount_scaler(coding_amount),
meetings_scaled = meeting_scaler(meetings),
work_week_scaled = work_week_scaler(work_week),
job_title_scaled = job_title_scaler.get(job_title),
etl_scaled = etl_scaler.get(etl_involvement),
setting_up_maintaining_data_platforms_scaled=data_platforms_scaler.get(setting_up_maintaining_data_platforms),
developing_prototype_models_scaled = developing_prototype_models_scaler.get(developing_prototype_models),
developing_hardware_scaled = developing_hardware_scaler.get(developing_hardware),
organizing_guiding_team_projects_scaled = organizing_guiding_team_projects_scaler.get(organizing_guiding_team_projects),
identifying_bus_analytics_problems_scaled = ibap_scaler.get(identifying_bus_analytics_problems),
communicating_people_outside_company_scaled = cpoc_scaler.get(communicating_people_outside_company),
most_work_done_with_cloud_computing_scaled = (3.2 if most_work_done_with_cloud_computing == 1 else 0),
languages_used_scaled = languages_used_scaler(languages_used.split(", ")),
tools_used_scaled = tools_used_scaler(tools_used.split(", ")))
```

# Results:


```python
results_df = pd.DataFrame.from_dict(results, orient='index', dtype=float)
for_display = results_df.apply(lambda x: "%g" % x, axis=1)
pd.DataFrame(for_display.sort_index())
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>academic_specialty_scaled</th>
      <td>3.9</td>
    </tr>
    <tr>
      <th>age_scaled</th>
      <td>22.5</td>
    </tr>
    <tr>
      <th>bargaining_skill_points_scaled</th>
      <td>37</td>
    </tr>
    <tr>
      <th>coding_amount_scaled</th>
      <td>-0.5</td>
    </tr>
    <tr>
      <th>communicating_people_outside_company_scaled</th>
      <td>5.4</td>
    </tr>
    <tr>
      <th>company_age_scaled</th>
      <td>-4.3</td>
    </tr>
    <tr>
      <th>company_size_scaled</th>
      <td>7.7</td>
    </tr>
    <tr>
      <th>constant</th>
      <td>60</td>
    </tr>
    <tr>
      <th>currently_student_scaled</th>
      <td>-0</td>
    </tr>
    <tr>
      <th>developing_hardware_scaled</th>
      <td>0</td>
    </tr>
    <tr>
      <th>developing_prototype_models_scaled</th>
      <td>12.1</td>
    </tr>
    <tr>
      <th>etl_scaled</th>
      <td>4.5</td>
    </tr>
    <tr>
      <th>gender_scaled</th>
      <td>0</td>
    </tr>
    <tr>
      <th>have_phd_scaled</th>
      <td>12.2</td>
    </tr>
    <tr>
      <th>identifying_bus_analytics_problems_scaled</th>
      <td>6.7</td>
    </tr>
    <tr>
      <th>industry_scaled</th>
      <td>2.2</td>
    </tr>
    <tr>
      <th>job_title_scaled</th>
      <td>20.2</td>
    </tr>
    <tr>
      <th>languages_used_scaled</th>
      <td>4.6</td>
    </tr>
    <tr>
      <th>meetings_scaled</th>
      <td>21.1</td>
    </tr>
    <tr>
      <th>most_work_done_with_cloud_computing_scaled</th>
      <td>3.2</td>
    </tr>
    <tr>
      <th>organizing_guiding_team_projects_scaled</th>
      <td>9.7</td>
    </tr>
    <tr>
      <th>setting_up_maintaining_data_platforms_scaled</th>
      <td>0</td>
    </tr>
    <tr>
      <th>state_gdp_scaled</th>
      <td>168.822</td>
    </tr>
    <tr>
      <th>tools_used_scaled</th>
      <td>15.7</td>
    </tr>
    <tr>
      <th>work_week_scaled</th>
      <td>0</td>
    </tr>
    <tr>
      <th>years_experience_scaled</th>
      <td>11.4</td>
    </tr>
  </tbody>
</table>
</div>




```python
predicted = results_df.apply(lambda x: sum(x) ** 2, axis=0)
print('According to the model, your predicted salary is {}'.format(locale.currency(predicted.ix[0,:], symbol=True, grouping=True)))
```

    According to the model, your predicted salary is $179,879.61



```python

```
