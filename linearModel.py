import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, neighbors
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype !=np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x=0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x = x+1

            df[column] = list(map(convert_to_int, df[column]))

    return df


def cleandata(df):

    df.loc[df['FTR'] == 'D', 'FTR'] = -1
    df.loc[df['FTR'] == 'H', 'FTR'] = 1
    df.loc[df['FTR'] == 'A', 'FTR'] = 0

    df.loc[df['HTR'] == 'D', 'HTR'] = -1
    df.loc[df['HTR'] == 'H', 'HTR'] = 1
    df.loc[df['HTR'] == 'A', 'HTR'] = 0

    df['Referee'] = 0
    df['Date'] = 0


df1 = pd.read_csv("/home/shadix/Football league Prediction/EPL Dataset/E0-2016.csv")
df = pd.read_csv("/home/shadix/Football league Prediction/EPL Dataset/E0-2017.csv")
df2 = pd.read_csv("/home/shadix/Football league Prediction/EPL Dataset/E0-2015.csv")
df3 = pd.read_csv("/home/shadix/Football league Prediction/EPL Dataset/E0-2014.csv")
df4 = pd.read_csv("/home/shadix/Football league Prediction/EPL Dataset/E0-2012.csv")
df5 = pd.read_csv("/home/shadix/Football league Prediction/EPL Dataset/E0-2011.csv")
df6 = pd.read_csv("/home/shadix/Football league Prediction/EPL Dataset/E0-2010.csv")
df7 = pd.read_csv("/home/shadix/Football league Prediction/EPL Dataset/E0-2009.csv")
df8 = pd.read_csv("/home/shadix/Football league Prediction/EPL Dataset/E0-2008.csv")
df9 = pd.read_csv("/home/shadix/Football league Prediction/EPL Dataset/E0-2007.csv")
df10 = pd.read_csv("/home/shadix/Football league Prediction/EPL Dataset/E0-2006.csv")
df11 = pd.read_csv("/home/shadix/Football league Prediction/EPL Dataset/E0-2005.csv")
df12 = pd.read_csv("/home/shadix/Football league Prediction/EPL Dataset/E0-2004.csv",error_bad_lines=False)
df13 = pd.read_csv("/home/shadix/Football league Prediction/EPL Dataset/E0-2003.csv",error_bad_lines=False)
df14 = pd.read_csv("/home/shadix/Football league Prediction/EPL Dataset/E0-2001.csv",error_bad_lines=False)
df15 = pd.read_csv("/home/shadix/Football league Prediction/EPL Dataset/E0-2000.csv",error_bad_lines=False)
df16 = pd.read_csv("/home/shadix/Football league Prediction/EPL Dataset/E0-1999.csv",error_bad_lines=False)
df17 = pd.read_csv("/home/shadix/Football league Prediction/EPL Dataset/E0-1998.csv",error_bad_lines=False)
df18 = pd.read_csv("/home/shadix/Football league Prediction/EPL Dataset/E0-1997.csv",error_bad_lines=False)
df19 = pd.read_csv("/home/shadix/Football league Prediction/EPL Dataset/E0-1996.csv",error_bad_lines=False)

frames = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17, df19]
bdf = pd.concat(frames, axis=0, ignore_index=True)

bdf = bdf.fillna(0)

temp = pd.read_csv("/home/shadix/Football league Prediction/EPL Dataset/E0-2017.csv")
cleandata(temp)

X = bdf['AR'].values
Y = bdf['FTAG'].values

mean_x = np.mean(X)
mean_y = np.mean(Y)

n = len(X)

numer = 0
denom = 0

for i in range(n):
    numer += (X[i] - mean_x)*(Y[i] - mean_y)
    denom += (X[i] - mean_x)**2

b1 = numer/denom
b0 = mean_y - (b1*mean_x)

print(b1, b0)

max_x = np.max(X) + 100
min_x = np.min(X) - 100

x = np.linspace(min_x, max_x, 1000)
y = b0 + b1*x

plt.plot(x, y, color='#58b970', label='Regression Line')

plt.scatter(X, Y, c='#ef5423', label='Scatter Plot')

plt.xlabel('Red Card')
plt.ylabel('Away Goals')
plt.legend()
plt.show()

ss_t = 0
ss_r = 0

for i in range(n):
    y_pred = b0 +b1*X[i]
    ss_t += (Y[i] - mean_y)**2
    ss_r += (Y[i] - y_pred)**2
r2 = 1 - (ss_r/ss_t)
print(r2)

X = X.reshape((n ,1))
reg = LinearRegression()
reg = reg.fit(X, Y)
y_pred = reg.predict(X)
r2_score = reg.score(X, Y)
print(r2_score)









