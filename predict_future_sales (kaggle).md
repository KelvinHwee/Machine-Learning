

```python
###   load the required libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

###   set some configurations for plots
plt.rcParams['figure.figsize'] = (12,10)
pd.set_option('display.max_columns', None)
```


```python
###   read in the data files
path = r'C:\Users\eight\Desktop\Kelvin HDD\3. Coursera\A. Projects\6. Predict future sales --- Kaggle\competitive-data-science-predict-future-sales'
os.chdir(path)

item_cat = pd.read_csv("item_categories.csv")
items = pd.read_csv("items.csv")
shops = pd.read_csv("shops.csv")
sales_train = pd.read_csv("sales_train.csv")
sales_test = pd.read_csv("test.csv")
df = sales_train.copy()
df.dtypes
```




    date               object
    date_block_num      int64
    shop_id             int64
    item_id             int64
    item_price        float64
    item_cnt_day      float64
    dtype: object




```python
###   take a look at a sample of the files
print(item_cat.head())
print(items.head())
print(shops.head())
print(sales_train.head())
print(sales_train.tail())
print(sales_test.head())
```

            item_category_name  item_category_id
    0  PC - Гарнитуры/Наушники                 0
    1         Аксессуары - PS2                 1
    2         Аксессуары - PS3                 2
    3         Аксессуары - PS4                 3
    4         Аксессуары - PSP                 4
                                               item_name  item_id  \
    0          ! ВО ВЛАСТИ НАВАЖДЕНИЯ (ПЛАСТ.)         D        0   
    1  !ABBYY FineReader 12 Professional Edition Full...        1   
    2      ***В ЛУЧАХ СЛАВЫ   (UNV)                    D        2   
    3    ***ГОЛУБАЯ ВОЛНА  (Univ)                      D        3   
    4        ***КОРОБКА (СТЕКЛО)                       D        4   
    
       item_category_id  
    0                40  
    1                76  
    2                40  
    3                40  
    4                40  
                            shop_name  shop_id
    0   !Якутск Орджоникидзе, 56 фран        0
    1   !Якутск ТЦ "Центральный" фран        1
    2                Адыгея ТЦ "Мега"        2
    3  Балашиха ТРК "Октябрь-Киномир"        3
    4        Волжский ТЦ "Волга Молл"        4
             date  date_block_num  shop_id  item_id  item_price  item_cnt_day
    0  02.01.2013               0       59    22154      999.00           1.0
    1  03.01.2013               0       25     2552      899.00           1.0
    2  05.01.2013               0       25     2552      899.00          -1.0
    3  06.01.2013               0       25     2554     1709.05           1.0
    4  15.01.2013               0       25     2555     1099.00           1.0
                   date  date_block_num  shop_id  item_id  item_price  \
    2935844  10.10.2015              33       25     7409       299.0   
    2935845  09.10.2015              33       25     7460       299.0   
    2935846  14.10.2015              33       25     7459       349.0   
    2935847  22.10.2015              33       25     7440       299.0   
    2935848  03.10.2015              33       25     7460       299.0   
    
             item_cnt_day  
    2935844           1.0  
    2935845           1.0  
    2935846           1.0  
    2935847           1.0  
    2935848           1.0  
       ID  shop_id  item_id
    0   0        5     5037
    1   1        5     5320
    2   2        5     5233
    3   3        5     5232
    4   4        5     5268
    


```python
# convert 'date' column into type "datetime"
from datetime import datetime
df.date = pd.to_datetime(df.date, format = "%d.%m.%Y")
df.dtypes
print(df.head(10))
```

            date  date_block_num  shop_id  item_id  item_price  item_cnt_day
    0 2013-01-02               0       59    22154      999.00           1.0
    1 2013-01-03               0       25     2552      899.00           1.0
    2 2013-01-05               0       25     2552      899.00          -1.0
    3 2013-01-06               0       25     2554     1709.05           1.0
    4 2013-01-15               0       25     2555     1099.00           1.0
    5 2013-01-10               0       25     2564      349.00           1.0
    6 2013-01-02               0       25     2565      549.00           1.0
    7 2013-01-04               0       25     2572      239.00           1.0
    8 2013-01-11               0       25     2572      299.00           1.0
    9 2013-01-03               0       25     2573      299.00           3.0
    


```python
###   create a function to obtain the month info
def create_period(df):
    
    get_mth = lambda x: x.strftime('%m')
    df['mth'] = df.date.apply(get_mth)
    df.mth = df.mth.astype('int64')
    
    #get_yr_mth = lambda x: x.strftime('%Y-%m')    
    #df['yr_mth'] = df.date.apply(get_yr_mth)    
    
    return df
```


```python
# create month info 
df_period = create_period(df)
df_period.shape
df_period.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>date_block_num</th>
      <th>shop_id</th>
      <th>item_id</th>
      <th>item_price</th>
      <th>item_cnt_day</th>
      <th>mth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2013-01-02</td>
      <td>0</td>
      <td>59</td>
      <td>22154</td>
      <td>999.00</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2013-01-03</td>
      <td>0</td>
      <td>25</td>
      <td>2552</td>
      <td>899.00</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2013-01-05</td>
      <td>0</td>
      <td>25</td>
      <td>2552</td>
      <td>899.00</td>
      <td>-1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013-01-06</td>
      <td>0</td>
      <td>25</td>
      <td>2554</td>
      <td>1709.05</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013-01-15</td>
      <td>0</td>
      <td>25</td>
      <td>2555</td>
      <td>1099.00</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
pivot_df = df_period.pivot_table(index = ['shop_id', 'item_id', 'mth'], 
								 values = 'item_cnt_day',
								 columns = 'date_block_num',
								 aggfunc = 'sum').fillna(0.0).reset_index()

pivot_df = pivot_df.groupby(['shop_id','item_id']).max().reset_index() ###
pivot_df.head()

pivot_df.shape
df2 = pivot_df.reset_index()
df2.head()
df2.shape
```




    (424124, 38)




```python
###   merge in all the relevant info into the earlier "df" for completeness and for ML
df3 = pd.merge(df2, items, how = "inner", on = "item_id")
df4 = pd.merge(df3, item_cat, how = "inner", on = "item_category_id")
df5 = pd.merge(df4, shops, how = "inner", on = "shop_id")
df5.head()
df5.shape
```




    (424124, 42)




```python
df5.isna().sum() # there are no NAs
```




    index                 0
    shop_id               0
    item_id               0
    mth                   0
    0                     0
    1                     0
    2                     0
    3                     0
    4                     0
    5                     0
    6                     0
    7                     0
    8                     0
    9                     0
    10                    0
    11                    0
    12                    0
    13                    0
    14                    0
    15                    0
    16                    0
    17                    0
    18                    0
    19                    0
    20                    0
    21                    0
    22                    0
    23                    0
    24                    0
    25                    0
    26                    0
    27                    0
    28                    0
    29                    0
    30                    0
    31                    0
    32                    0
    33                    0
    item_name             0
    item_category_id      0
    item_category_name    0
    shop_name             0
    dtype: int64




```python
# we change the order of the columns
[(i, df5.columns[i]) for i in range(len(df5.columns))]
col_order = [0,1,41,2,38,39,40,3] + list(range(4,38))
df6 = df5.iloc[:,col_order]
df6.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>shop_id</th>
      <th>shop_name</th>
      <th>item_id</th>
      <th>item_name</th>
      <th>item_category_id</th>
      <th>item_category_name</th>
      <th>mth</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
      <th>26</th>
      <th>27</th>
      <th>28</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
      <th>32</th>
      <th>33</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>!Якутск Орджоникидзе, 56 фран</td>
      <td>30</td>
      <td>007: КООРДИНАТЫ «СКАЙФОЛЛ»</td>
      <td>40</td>
      <td>Кино - DVD</td>
      <td>2</td>
      <td>0.0</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>!Якутск Орджоникидзе, 56 фран</td>
      <td>32</td>
      <td>1+1</td>
      <td>40</td>
      <td>Кино - DVD</td>
      <td>2</td>
      <td>6.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>0</td>
      <td>!Якутск Орджоникидзе, 56 фран</td>
      <td>35</td>
      <td>10 ЛЕТ СПУСТЯ</td>
      <td>40</td>
      <td>Кино - DVD</td>
      <td>2</td>
      <td>1.0</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>0</td>
      <td>!Якутск Орджоникидзе, 56 фран</td>
      <td>43</td>
      <td>100 МИЛЛИОНОВ ЕВРО</td>
      <td>40</td>
      <td>Кино - DVD</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14</td>
      <td>0</td>
      <td>!Якутск Орджоникидзе, 56 фран</td>
      <td>75</td>
      <td>12 ДРУЗЕЙ ОУШЕНА WB (регион)</td>
      <td>40</td>
      <td>Кино - DVD</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
###   create some visualisations
df6.hist()
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x0000024BDDD70860>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000024BDDCD6BA8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB4EE80F0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000024BE32064E0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB3D10A90>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB3F55080>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB3F78630>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB4E73C18>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB4E73C50>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB3F15780>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB3E96D30>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB3E53320>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB4E418D0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB3DAEE80>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB3D74470>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB3DFAA20>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB3EA4FD0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB3D5C5C0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x0000024CB662B4A8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB4234160>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB4263710>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB4296CC0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB42D42B0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB4302860>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB4336E10>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB4373400>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB43A39B0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB4AA7F60>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB4AE3550>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB4B15B00>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB4B530F0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB4B846A0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB4BB4C50>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB4BF2240>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB4C227F0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB4C55DA0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB4C92390>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB62E2940>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB6315EF0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB63524E0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB6382A90>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000024BB63C0080>]],
          dtype=object)




![png](output_10_1.png)



```python
###############################################################################
###   create a machine learning model
###############################################################################

# using all data to predict 'item_price' given that 'date_block_num' is 34
# we insert the price info as a row, mth as 11, 
# then use the predicted item price to predict the item count for the month
X = df6.loc[:,df6.columns != 33]
X = X.drop(['item_name', 'item_category_name', 'shop_name'], axis = 1)
y = df5.loc[:,df5.columns == 33]
print(X.head())
print(y.head())

# create an XGBoost model
import xgboost as xgb
data_dmatrix = xgb.DMatrix(data = X, label = y)

# do a train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
xg_reg = xgb.XGBRegressor(num_round = 1000, 
                          verbosity = 0, # silent all warning messages                          
                          eval_metric = 'rmse',                          
                          min_child_weight = 0.5, 
                          eta = 0.2, # something like learning rate
                          seed = 100,                          
                          gamma = 10, # min loss reduction to make a further partition
                          max_depth = 8, # increasing this value makes model more complex
                          n_estimators = 100)

xg_reg.fit(X_train, y_train)
```

       index  shop_id  item_id  item_category_id  mth    0     1    2    3    4  \
    0      0        0       30                40    2  0.0  31.0  0.0  0.0  0.0   
    1      2        0       32                40    2  6.0  10.0  0.0  0.0  0.0   
    2      4        0       35                40    2  1.0  14.0  0.0  0.0  0.0   
    3      8        0       43                40    1  1.0   0.0  0.0  0.0  0.0   
    4     14        0       75                40    1  1.0   0.0  0.0  0.0  0.0   
    
         5    6    7    8    9   10   11   12   13   14   15   16   17   18   19  \
    0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   
    1  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   
    2  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   
    3  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   
    4  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   
    
        20   21   22   23   24   25   26   27   28   29   30   31   32  
    0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  
    1  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  
    2  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  
    3  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  
    4  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  
        33
    0  0.0
    1  0.0
    2  0.0
    3  0.0
    4  0.0
    




    XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                 colsample_bynode=1, colsample_bytree=1, eta=0.2,
                 eval_metric='rmse', gamma=10, importance_type='gain',
                 learning_rate=0.1, max_delta_step=0, max_depth=8,
                 min_child_weight=0.5, missing=None, n_estimators=100, n_jobs=1,
                 nthread=None, num_round=1000, objective='reg:linear',
                 random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 seed=100, silent=None, subsample=1, verbosity=0)




```python
y_preds = xg_reg.predict(X_test)
print(y_preds)
```

    [-0.01624572  0.0297519  -0.00831491 ...  1.6636343  -0.00496095
     -0.00663596]
    


```python
# we measure the accuracy using RMSE
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, pd.DataFrame(y_preds)))
print(rmse)
```

    2.001824470229297
    


```python
###############################################################################
###   do the final prediction
###############################################################################

# this is based on the merged data with the 'sales_test' dataframe
test_df = sales_test.merge(df6, how = 'left', on = ['shop_id', 'item_id']).fillna(0.0)

# reorder the columns again, and dropping less important columns
[(i,test_df.columns[i]) for i in range(len(test_df.columns))]
col_order_test = [3,1,2,6,8] + list(range(9,43))
test_df = test_df.iloc[:,col_order_test]

# previously we fitted the model without one column, 
# so, now we want to predict the 34th column, so we move one month forward
# i.e. from 1 to 33 (including renaming the column names)
dic_names = dict(zip(test_df.columns[5:], list(np.array(list(test_df.columns[5:])) - 1)))
test_df = test_df.rename(dic_names, axis = 1)

test_df_select = test_df.iloc[:,test_df.columns != -1]
test_df_select.head()

# the final prediction
y_preds_test = xg_reg.predict(test_df_select)
```


```python
# we clip the predictions to a range between 0 to 20
clipped_preds = list(map(lambda x: min(20, max(x,0)), list(y_preds_test)))
results = pd.DataFrame({'ID': test_df.index, 'item_cnt_month': clipped_preds})

results.head(20)
results.shape
```


```python
# read into the submission file
results.describe()
results.to_csv('try.csv')
```
