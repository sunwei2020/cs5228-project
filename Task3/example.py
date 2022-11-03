import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

df = pd.read_csv("../data/train.csv")
df = df.loc[df['built_year'].notnull()]
# Count the number of different built_year
dict= df.groupby(['built_year'])['listing_id'].count().to_dict()
df['count'] = df['built_year'].map(dict)

#Set planning_area we are going to prodict
target1='clementi'
df1=df.loc[df['planning_area']==target1]

target2='yishun'
df2=df.loc[df['planning_area']==target2]

target3='queenstown'
df3=df.loc[df['planning_area']==target3]

target4='orchard'
df4=df.loc[df['planning_area']==target4]

target5='jurong east'
df5=df.loc[df['planning_area']==target5]

target6='jurong west'
df6=df.loc[df['planning_area']==target6]

df1 = df1[['built_year','count']].rename(columns = {"built_year":"ds","count":"y"})
df1.drop_duplicates(inplace=True)
df1['ds'] = pd.to_datetime(df1['ds'],format="%Y") #transfer the built_year into standard time

df2 = df2[['built_year','count']].rename(columns = {"built_year":"ds","count":"y"})
df2.drop_duplicates(inplace=True)
df2['ds'] = pd.to_datetime(df2['ds'],format="%Y") #transfer the built_year into standard time

df3 = df3[['built_year','count']].rename(columns = {"built_year":"ds","count":"y"})
df3.drop_duplicates(inplace=True)
df3['ds'] = pd.to_datetime(df3['ds'],format="%Y") #transfer the built_year into standard time

df4 = df4[['built_year','count']].rename(columns = {"built_year":"ds","count":"y"})
df4.drop_duplicates(inplace=True)
df4['ds'] = pd.to_datetime(df4['ds'],format="%Y") #transfer the built_year into standard time

df5 = df5[['built_year','count']].rename(columns = {"built_year":"ds","count":"y"})
df5.drop_duplicates(inplace=True)
df5['ds'] = pd.to_datetime(df5['ds'],format="%Y") #transfer the built_year into standard time

df6 = df6[['built_year','count']].rename(columns = {"built_year":"ds","count":"y"})
df6.drop_duplicates(inplace=True)
df6['ds'] = pd.to_datetime(df6['ds'],format="%Y") #transfer the built_year into standard time


m1 = Prophet(changepoint_prior_scale=0.8, interval_width=0.3,yearly_seasonality=12)
m1.fit(df1)

m2 = Prophet(changepoint_prior_scale=0.8, interval_width=0.3,yearly_seasonality=12)
m2.fit(df2)
m3 = Prophet(changepoint_prior_scale=0.8, interval_width=0.3,yearly_seasonality=12)
m3.fit(df3)
m4 = Prophet(changepoint_prior_scale=0.8, interval_width=0.3,yearly_seasonality=12)
m4.fit(df4)
m5 = Prophet(changepoint_prior_scale=0.8, interval_width=0.3,yearly_seasonality=12)
m5.fit(df5)
m6 = Prophet(changepoint_prior_scale=0.8, interval_width=0.3,yearly_seasonality=12)
m6.fit(df6)

future1 = m1.make_future_dataframe(periods=5, freq='Y') # range of prediction
prediction1 = m1.predict(future1)
prediction1['yhat_lower']=0 # restric the minmum of predict



future2 = m2.make_future_dataframe(periods=8, freq='Y') # range of prediction
prediction2 = m2.predict(future2)
prediction2['yhat_lower']=0 # restric the minmum of predict
future3 = m3.make_future_dataframe(periods=8, freq='Y') # range of prediction
prediction3 = m3.predict(future3)
prediction3['yhat_lower']=0 # restric the minmum of predict

future4 = m4.make_future_dataframe(periods=8, freq='Y') # range of prediction
prediction4 = m4.predict(future4)
prediction4['yhat_lower']=0 # restric the minmum of predict

future5 = m5.make_future_dataframe(periods=8, freq='Y') # range of prediction
prediction5 = m5.predict(future5)
prediction5['yhat_lower']=0 # restric the minmum of predict
future6 = m6.make_future_dataframe(periods=8, freq='Y') # range of prediction
prediction6 = m6.predict(future6)
prediction6['yhat_lower']=0 # restric the minmum of predict

prediction1=prediction1.loc[prediction1['ds']>'2000-01-01']
prediction2=prediction2.loc[prediction2['ds']>'2000-01-01']
prediction3=prediction3.loc[prediction3['ds']>'2000-01-01']
prediction4=prediction4.loc[prediction4['ds']>'2000-01-01']
prediction5=prediction5.loc[prediction5['ds']>'2000-01-01']
prediction6=prediction6.loc[prediction6['ds']>'2000-01-01']
# # Format the chart
# m1.plot(prediction1)
# m2.plot(prediction2)
# m3.plot(prediction3)
# m3.plot(prediction3)


ax = plt.subplots() # 创建图实例
ax=plt.gca()
ax.get_yaxis().get_major_formatter().set_scientific(False)
ax.plot(prediction1['ds'], prediction1['yhat'], label=target1)
ax.plot(prediction2['ds'], prediction2['yhat'], label=target2)
ax.plot(prediction3['ds'], prediction3['yhat'], label=target3)
ax.plot(prediction4['ds'], prediction4['yhat'], label=target4)
ax.plot(prediction5['ds'], prediction5['yhat'], label=target5)
ax.plot(prediction6['ds'], prediction6['yhat'], label=target6)

ax.set_xlabel("Built Year") #设置x轴名称 x label
ax.set_ylabel("Number for sale") #设置y轴名称 y label
ax.set_title('Prediction') #设置图名为Simple Plot
ax.legend() #自动检测要在图例中显示的元素，并且显示
plt.show() #图形可视化
