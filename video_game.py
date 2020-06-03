#!/usr/bin/env python
# coding: utf-8

# # 2.2 Video Game Sales 电子游戏销售分析
# 电子游戏市场分析：受欢迎的游戏、类型、发布平台、发行人等；
# 预测每年电子游戏销售额。
# 可视化应用：如何完整清晰地展示这个销售故事。
# 代码仓库：https://github.com/Graceqi/video_game
# 数据集：https://www.kaggle.com/gregorut/videogamesales

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from itertools import combinations
from scipy.stats import pearsonr,spearmanr,kendalltau
plt.rc("font",family="SimHei",size="15")  #解决中文乱码问题


# In[2]:


video_data = pd.read_csv(r"D:\数据挖掘课件视频\数据集\vgsales\vgsales.csv")
print(video_data.columns)


# In[3]:


video_data.head()


# In[4]:


# 统计缺失值数量
missing=video_data.isnull().sum().reset_index().rename(columns={0:'missNum'})
# 计算缺失比例
missing['missRate']=missing['missNum']/video_data.shape[0]
# 按照缺失率排序显示
miss_analy=missing[missing.missRate>=0].sort_values(by='missRate',ascending=False)
print(miss_analy)
# miss_analy 存储的是每个变量缺失情况的数据框
fig = plt.figure(figsize=(18,6))
plt.bar(np.arange(miss_analy.shape[0]), list(miss_analy.missRate.values), align = 'center',color=['red','green','yellow','steelblue'])

plt.title('Histogram of missing value of variables')
plt.xlabel('variables names')
plt.ylabel('missing rate')
# 添加x轴标签，并旋转90度
plt.xticks(np.arange(miss_analy.shape[0]),list(miss_analy['index']))
plt.xticks(rotation=90)
# 添加数值显示
for x,y in enumerate(list(miss_analy.missRate.values)):
    plt.text(x,y+0.12,'{:.2%}'.format(y),ha='center',rotation=90)    
plt.ylim([0,1.2])
    
plt.show()


# In[5]:


#去掉缺失值所在行
print(type(video_data))
video_data=video_data.dropna()


# # 一、通过统计平台、类别、出版商的频率分布，来发现拥有最多游戏的游戏平台、游戏最多类别的、出版最多游戏的出版商

# In[6]:


Platform=video_data['Platform'].value_counts()
Genre=video_data['Genre'].value_counts()
Publisher=video_data['Publisher'].value_counts()
print("Platform\n",Platform)
print("Genre\n",Genre)
print("Publisher\n",Publisher)


# ## 拥有最多游戏的游戏平台是DS、游戏最多类别的是Action、出版最多游戏的出版商是Electronic Arts

# ## 通过可视化数据的频率分布，来直观分析出有最多游戏的平台、游戏最常见的类别，以及出品游戏最多的出版商

# ## 对标称属性的分布进行统计分析Platform,Genre,Publisher

# In[7]:


list_nominal=[video_data.Platform,video_data.Genre]
list_name=['Platform','Genre']
fig = plt.figure(figsize=(12,5))
for d,i in zip(list_nominal,range(3)):
    plt.subplot(1,2,i+1)
    plt.xlabel(f"{list_name[i]}")
    plt.hist(x = d, # 指定绘图数据
         bins = 50, # 指定直方图中条块的个数
         color = 'steelblue', # 指定直方图的填充色
         edgecolor = 'black' # 指定直方图的边框色
         )
    plt.title(f"The Frequency Distribution of {list_name[i]}")
    plt.xticks(rotation=90)
    plt.tight_layout()
plt.show()


# 最常见的平台是DS和PS2,最常见的游戏类别是Action，其次就是Sports

# In[8]:


Publisher=video_data['Publisher'].value_counts()
Publisher[:25].plot.bar(figsize=(6,2))
# 添加x轴和y轴标签
plt.xlabel('Publisher')
plt.ylabel('Frequency Distribution')
# 添加标题
plt.title('The Frequency Distribution of Publisher')
plt.show()


# 出版最多游戏的出版商是Electronic Arts

# # 二、统计分析游戏最高产的年份，以及游戏在不同地区的销量对比

# ## 对数值属性Year,NA_Sales,EU_Sales,JP_Sales,Other_Sales,Global_Sales进行分析

# In[9]:


fig = plt.figure(figsize=(6,3))
sns.distplot(video_data.Year,bins = 10,hist = True,kde = True,rug = True,norm_hist=False,color = 'y',label = 'distplot',axlabel = 'Year')
plt.title("The Frequency Distribution of Year")
plt.show()


# ## 在2010年左右，游戏是发行高峰

# ## 画出NA_Sales,EU_Sales,JP_Sales,Other_Sales,Global_Sales的分布图和盒图

# In[10]:


list_nominal=[video_data.NA_Sales,video_data.EU_Sales,video_data.JP_Sales,video_data.Other_Sales,video_data.Global_Sales]
list_name=[ 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
fig = plt.figure(figsize=(12,10))
for d,i in zip(list_nominal,range(6)):
    plt.subplot(2,3,i+1)
    plt.xlabel(f"{list_name[i]}")
    sns.distplot(d)
#     plt.tight_layout()
#     #std标准差
    print(f"{list_name[i]}:")
    print(d.describe())
#     #skewness and kurtosis偏度和峰度
#     print("Skewness: %f" % d.skew())
#     print("Kurtosis: %f" % d.kurt())
plt.show()


# 由上面的频率分布图和下面的盒图可以看出，NA_Sales,EU_Sales,JP_Sales,Other_Sales,Global_Sales这五列数据的分布比较离散化。并且接近于0的数据较多

# In[11]:


ax = sns.boxplot(data=video_data[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']], orient="h", palette="Set2")


# ## 计算'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'之间的相似度:

# In[12]:


data=list(combinations(video_data[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']], 2))
print(data)


# In[13]:


for d in data:
    print(d)
    x=video_data[d[0]]
    y=video_data[d[1]]
    #皮尔森相似度
    print("皮尔森相似度",pearsonr(x,y)[0])
    #余弦相似度计算方法
    tmp=sum(a*b for a,b in zip(x,y))
    non=np.linalg.norm(x)*np.linalg.norm(y)
    print("余弦相似度",round(tmp/float(non),3))
    #欧几里得相似度计算方法
    print("欧几里得相似度计算",math.sqrt(sum(pow(a-b,2) for a,b in zip(x,y))))


# ## 由上面的相似度计算可以得出，'NA_Sales', 'EU_Sales'分别对总收入'Global_Sales'的相似度较高，说明'NA_Sales', 'EU_Sales'占总收入'Global_Sales'的比重较大， 而'JP_Sales', 'Other_Sales'对总收入的相似度较低，说明所占比重较小。而且'NA_Sales', 'EU_Sales'，'Global_Sales'之间的相似度都在0.7以上，唯独'JP_Sales'与其他三项的相似度均在0.4，说明日本的销量'JP_Sales'与其他国家销量的关系不大。

# # 三、通过对不同游戏的名称、平台、类别、出版商分别来计算全球销量，来统计出最受欢迎的游戏、类型、发布平台、发行人

# In[14]:


Name_data=video_data.groupby(by=['Name'])['Global_Sales'].sum()
Name_data.sort_values(ascending = False, inplace = True)
Name_data


# ## 通过对游戏名称统计并排序总销量，得出Wii Sports的游戏销量最高，达到82.74 million的全球销量，最受欢迎。

# In[15]:


# Index(['Rank', 'Name', 'Platform', 'Year', 'Genre', 'Publisher', 'NA_Sales',
#        'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'],
#       dtype='object')

Genre_data=video_data.groupby(by=['Genre'])['Global_Sales'].sum()
Genre_data.sort_values(ascending = False, inplace = True)
Genre_data


# ## 通过对游戏类别统计并排序总销量，得出Action类别的游戏销量最高，全球总销量高达1722.84 million最受欢迎。

# In[16]:


Platform=video_data.groupby(by=['Platform'])['Global_Sales'].sum()
Platform.sort_values(ascending = False, inplace = True)
Platform


# ## 通过对游戏平台统计并排序总销量，得出PS2平台的游戏销量最高，全球总销量高达1233.46 million最受欢迎。

# In[17]:


Publisher=video_data.groupby(by=['Publisher'])['Global_Sales'].sum()
Publisher.sort_values(ascending = False, inplace = True)
Publisher


# ## 通过对游戏出版商统计并排序总销量，得出Nintendo出版商出版的游戏总销量最高，全球总销量高达1784.43 million最受欢迎。

# ## 经过统计得出结论：销量最高也就是最受欢迎的游戏是Wii Sports 、类型是Action、发布平台是PS2、发行人是Nintendo

# # 四、 由于大多数游戏只有某一年的销售量，没有足够的数据来对每一个做销量预测。因此，可以按年份统计所有销售数据，来对总销售额做预测。

# In[18]:


Year=video_data.groupby(by=['Year'])['Global_Sales'].sum()
# Year.sort_values(ascending = False, inplace = True)
Year.sort_index()
dict_Year= {'Year':Year.index,'Sum':Year.values}
df_Year = pd.DataFrame(dict_Year)
print(df_Year)


# In[19]:


fig = plt.figure(figsize=(6,3))
# sns.distplot(df_Year.Year,bins = 10,hist = True,kde = True,rug = True,norm_hist=False,color = 'y',label = 'distplot',axlabel = 'Year')
sns.lineplot(x=df_Year.Year, y=df_Year.Sum,err_style="bars", ci=68, data=df_Year)
plt.title("每年游戏全球总销量分布图")
plt.show()


# ## 由上图可看出，每年游戏总销量的分布图与之前每年游戏数量的分布图很相似

# In[20]:


video_data.Global_Sales.describe()


# In[21]:


sns.boxplot(data=video_data[['Global_Sales']], orient="h", palette="Set2")


# In[22]:


num=video_data[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']]
corr = num.corr()

fig = plt.figure(figsize=(15,10))

#Here we use cmap CoolWarm as it gives us a better view of postive and negative correlation.
#And with the help of vmin and vmax set to -1 and +1 , the features having values closer to +1 have positive correlation and features having values closer to -1 have negative correlation.
sns.heatmap(corr,annot=True,linewidths=.5,cmap='coolwarm',vmin=-1,vmax=1,center=0);


# In[23]:


from sklearn.model_selection import train_test_split as tts
X=video_data[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
y=video_data[[ 'Global_Sales']]
X_train,X_test,y_train,y_test=tts(X,y,test_size=0.3,random_state=0)


# In[24]:


from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

feature_sel_model = SelectFromModel(Lasso(alpha=0.005, random_state=0)) 
feature_sel_model.fit(X_train, y_train)


# In[25]:


selected_feat = X_train.columns[(feature_sel_model.get_support())]
print(selected_feat)


# In[26]:


X_train = X_train[selected_feat].reset_index(drop=True)


# In[27]:


X_train.head()


# In[28]:


X_test=X_test[selected_feat]


# In[29]:


X_test.head()


# # 线性回归做销量预测

# In[30]:


from sklearn.linear_model import LinearRegression
from sklearn import metrics

lm = LinearRegression()

#Fitting linear model on train dataset
lm.fit(X_train,y_train)

#Test dataset prediction
lm_predictions = lm.predict(X_test)

#Scatterplot
plt.scatter(y_test, lm_predictions)
plt.show()

#Evaluation
print("MAE:", metrics.mean_absolute_error(y_test, lm_predictions))
print('MSE:', metrics.mean_squared_error(y_test, lm_predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lm_predictions)))

#Accuracy
print("\nAccuracy : {}".format(lm.score(X_test,y_test)))


# ## 准确率高达0.9999856748617503
