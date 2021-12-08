# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 16:59:54 2021

@author: Win10
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
import seaborn as sns

    # 讀取資料檔案
readPath = "C:\\Users\\Win10\\Desktop\\class\\小組作業\\student\\student-mat.csv"  # 資料檔案的地址和檔名
dfOpenFile = pd.read_csv(readPath, header=0, sep=";")  # 間隔符為逗號，首行為標題行
dfData = dfOpenFile.dropna()  # 刪除含有缺失值的資料
print(dfData.keys())
#Index(['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
#       'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
#       'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
#       'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
#       'Walc', 'health', 'absences', 'G1', 'G2', 'G3'],
#      dtype='object')

dfData.shape
#(395, 33)

dfData.head(3)
#  school sex  age address famsize Pstatus  ...  Walc  health absences G1 G2  G3
#0     GP   F   18       U     GT3       A  ...     1       3        6  5  6   6
#1     GP   F   17       U     GT3       T  ...     1       3        4  5  5   6
#2     GP   F   15       U     LE3       T  ...     3       3       10  7  8  10
#
#[3 rows x 33 columns]


#敘述性統計
dfData.describe()
#              age        Medu        Fedu  ...          G1          G2          G3
#count  395.000000  395.000000  395.000000  ...  395.000000  395.000000  395.000000
#mean    16.696203    2.749367    2.521519  ...   10.908861   10.713924   10.415190
#std      1.276043    1.094735    1.088201  ...    3.319195    3.761505    4.581443
#min     15.000000    0.000000    0.000000  ...    3.000000    0.000000    0.000000
#25%     16.000000    2.000000    2.000000  ...    8.000000    9.000000    8.000000
#50%     17.000000    3.000000    2.000000  ...   11.000000   11.000000   11.000000
#75%     18.000000    4.000000    3.000000  ...   13.000000   13.000000   14.000000
#max     22.000000    4.000000    4.000000  ...   19.000000   19.000000   20.000000

#[8 rows x 16 columns]

# 用 seaborn一次把圖表的美化格式設定好，這裡是只有先設定圖表長寬
sns.set(rc={'figure.figsize':(10,10)})
# 使用的資料是期末考成績  約呈常態分布
sns.distplot(dfData['G3'])
plt.show()

dfData['G3'].max() # 20
dfData['G3'].min() # 0
dfData['G3'].std() # 4.5814426109978434

# In[1]heatmap 相關係數 corr()
# 接下來我們可以看每個變數之間的關係
# 透過相關係數去觀察有哪些特徵變數和目標變數有較高的相關性等等：
correlation_matrix = dfData.corr().round(2)

# annot = True 讓我們可以把數字標進每個格子裡
sns.heatmap(data=correlation_matrix, annot = True)
#可知G1、G2變數與G3變數有最高的相關性
# In[2] G1、G2兩個變數跟G3變數的關係圖
# 用G1和G2來做出預測G3的模型。再次把這兩個變數跟G3變數的關係畫出來，可以看到兩者和G3變數都接近線性關係：

# G1 - first period grade (numeric: from 0 to 20)
# G2 - second period grade (numeric: from 0 to 20)

# 設定整張圖的長寬
plt.figure(figsize=(20, 5))
features = ['G1', 'G2']
target = dfData['G3']
for i, col in enumerate(features):
 # 排版1 row, 2 columns, nth plot：在jupyter notebook上兩張並排 
 plt.subplot(1, len(features) , i+1)
 # add data column into plot
 x = dfData[col]
 y = target
 plt.scatter(x, y, facecolor='xkcd:azure', edgecolor='black', s=20)
 plt.title(col)
 plt.xlabel(col)
 plt.ylabel('G3')

# 可以看到兩者 G1和G2 和 G3 變數都接近線性關係
# G1,G2和G3都是正向關係

# In[3] 準備模型的訓練資料(多元線性迴歸) 
# 用np.c_把G1和G2兩個欄位合併在一起，assign成X，把G3欄位assign成Y：
X = pd.DataFrame(np.c_[dfData['G1'], dfData['G2']], columns = ['G1','G2'])

y = dfData['G3']

# In[4] 產生模型
# 建出一個LinearRegression的物件後，用特徵變數的訓練資料和目標變數的訓練資料產生一個模型。接著將特徵變數的測試資料倒進這個新產生的模型當中，得到預測的目標變數資料。
# 最後將這個預測的目標變數資料（預測結果）和目標變數的測試資料（真實結果）做R2-score：

# Modeling
reg = LinearRegression()
# 學習/訓練Fitting linear model
reg.fit(X,y)
# 預測結果Predicting using the linear model
yFit=reg.predict(X)
# 真實結果：Y_test
# 測試準確度：

print("\nModel1: Y = b0 + b1*x1 + b2*x2")
print('迴歸截距: w0={}'.format(reg.intercept_))  # w0: 截距
print('迴歸係數: w1={}'.format(reg.coef_))  # w1,..wm: 迴歸係數
print('R2 確定係數：{:.4f}'.format(reg.score(X, y)))  # R2 判定係數
print('均方誤差：{:.4f}'.format(mean_squared_error(y, yFit)))  # MSE 均方誤差
print('平均絕對值誤差：{:.4f}'.format(mean_absolute_error(y, yFit)))  # MAE 平均絕對誤差
print('中位絕對值誤差：{:.4f}'.format(median_absolute_error(y, yFit)))  # 中值絕對誤差

# 輸出迴歸結果
#Model1: Y = b0 + b1*x1 + b2*x2
#迴歸截距: w0=-1.8300121405807346
#迴歸係數: w1=[0.15326859 0.98686684]
# R2:  0.8221632333156184
#均方誤差：3.7233
#平均絕對值誤差：1.1375
#中位絕對值誤差：0.7089

# In[6] 結論
# 我們用G1（期初成績）和G2（期中成績）藉由多元線性迴歸預測G3（期末成績）。
# 在其他變數保持不變下，當G1（期初成績）增加1 unit，G3（期末成績）就會大約增加0.15 unit。
# 同樣地，當G2（期中成績）增加1 unit，G3（期末成績）就會大約上升0.98 unit。看來期初成績與期中成績對於期末成績皆有正向影響，但比起期初成績，期中成績考得越好對於期末成績有更多的影響力。
# 得到的這個R2-score讓我們可以知道特徵變數對於目標變數的解釋程度為何，而越接近1代表越準確。這裡大約是66%，解釋程度算是相當好的。
# In[5] 預測的目標變數資料和測試的目標變數資料畫成散佈圖
# 如果我們把剛剛的預測的目標變數資料和測試的目標變數資料畫成散佈圖，可以看到兩者關係接近斜直線1：

# plotting the y_test vs y_pred
yFit=reg.predict(X)
plt.scatter(yFit, y)
plt.xlabel('Y_pred')
plt.ylabel('y')
plt.show()


# In[6-1 其他圖例]: 簡單線性迴歸 G1 VS G3
dfData.loc[:, ['G1']]
dfData.loc[:, ['G3']]
       
xG1, y = dfData.loc[:, ['G1']], dfData.loc[:, ['G3']]

reg.fit(xG1, y)

print('w_1 =', reg.coef_[0])
# w_1 = [1.10625609]
print('w_0 =', reg.intercept_)
# w_0 = [-1.65280383]

plt.scatter(xG1, y, facecolor='xkcd:azure', edgecolor='black', s=20)
plt.xlabel('G1', fontsize=14)
plt.ylabel("G1", fontsize=14)
n_xG1 = np.linspace(xG1.min(), xG1.max(), 100)
n_yG1 = reg.intercept_ + reg.coef_[0] * n_xG1
plt.plot(n_xG1, n_yG1, color='r', lw=3);

# In[6-2 其他圖例]: 簡單線性迴歸 G2 VS G3
dfData.loc[:, ['G2']]
dfData.loc[:, ['G3']]
       
xG2, y = dfData.loc[:, ['G2']], dfData.loc[:, ['G3']]

reg.fit(xG2, y)

print('w_1 =', reg.coef_[0])
# w_1 = [1.10211236]
print('w_0 =', reg.intercept_)
# w_0 = [-1.39275821]

plt.scatter(xG2, y, facecolor='xkcd:azure', edgecolor='black', s=20)
plt.xlabel('G2', fontsize=14)
plt.ylabel("G3", fontsize=14)
n_xG2 = np.linspace(xG2.min(), xG2.max(), 100)
n_yG2 = reg.intercept_ + reg.coef_[0] * n_xG2
plt.plot(n_xG2, n_yG2, color='r', lw=3);
# In[7-1 其他圖例]: 簡單線性迴歸 資料標準化  StandardScaler
from sklearn.preprocessing import StandardScaler

xStr = StandardScaler().fit_transform(xG2)
yStr = StandardScaler().fit_transform(y)

reg.fit(xStr, yStr)
print('w_1 =', reg.coef_[0])
# w_1 = [0.90486799]

print('w_0 =', reg.intercept_)
# w_0 = [-4.96770652e-17]

plt.scatter(xStr, yStr, facecolor='xkcd:azure', edgecolor='black', s=20)
plt.xlabel('G2', fontsize=14)
plt.ylabel("G3", fontsize=14)
# 繪製迴歸線
n_xStr = np.linspace(xStr.min(), xStr.max(), 100)
n_yStr = reg.intercept_ + reg.coef_[0] * n_xStr
plt.plot(n_xStr, n_yStr, color='r', lw=3)
plt.show()


# 透過 seaborn 可更簡單地繪製迴歸線
import seaborn as sns

# 預設會在迴歸線旁繪製 95% 的信賴區間
sns.regplot(x='G1', y='G3', data=dfData, 
            scatter_kws={'facecolor':'xkcd:azure', 
                         'edgecolor':'black', 's':20},
            line_kws={'color':'r', 'lw':3})

# 預設會在迴歸線旁繪製 95% 的信賴區間
sns.regplot(x='G2', y='G3', data=dfData, 
            scatter_kws={'facecolor':'xkcd:azure', 
                         'edgecolor':'black', 's':20},
            line_kws={'color':'r', 'lw':3})
# In[7-2 其他圖例]: python 線性迴歸 

sns.set(style="ticks")

sns.lmplot(x="G1", y="G3", col="G2", hue="G2", data=dfData,
           col_wrap=2, ci=None, palette="muted", height=4,
           scatter_kws={"s": 50, "alpha": 1})
