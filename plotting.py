import numpy as np
import pandas as pd
from sklearn import preprocessing, neighbors, tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
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
bdf['AwaySaves'] = bdf['HST'] - bdf['FTHG']
bdf['HomeSaves'] = bdf['AST'] - bdf['FTAG']

temp = pd.read_csv("/home/shadix/Football league Prediction/EPL Dataset/E0-2017.csv")
cleandata(temp)

team = raw_input("Team1: ")
team2 = raw_input("Team2: ")

temp['HomeSaves'] = temp['HST'] - temp['FTHG']
temp['AwaySaves'] = temp['AST'] - temp['FTAG']

ht = temp.loc[temp['HomeTeam'] == team]
at = ht.loc[ht['AwayTeam'] == team2]

at = at.fillna(0)

pre = at[['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HTR', 'HST', 'AST']].values
hsave = at[['HST']].values - at[['FTHG']].values
asave = at[['AST']].values - at[['FTAG']].values

print(hsave, asave)

print(pre)

bdf.drop(['Date'], 1, inplace=True)
cleandata(bdf)

X = bdf[['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HTR', 'HST', 'AST']].values
y = np.array(bdf['FTR'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6)

counta = 0
countd = 0
counth = 0

mean_acc = 0

for k in range(1, 91):
    clf = RandomForestClassifier(max_depth=k, oob_score=True, n_estimators=k)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(accuracy)
    mean_acc = mean_acc + accuracy
    print(" ")
    taccuracy = clf.score(X_train, y_train)
    print(taccuracy)
    print(" ")

    pre_val = clf.predict(pre)

    if pre_val == 1:
        counth += 1
    elif pre_val == -1:
        countd += 1
    elif pre_val == 0:
        counta += 1

    print(pre_val)

print(counth)
print(counta)
print(countd)

if counta >= counth:
    if counta > countd:
        print(team2)
    elif counta < countd:
        print("Draw")
elif counta <= counth:
    if counth > countd:
        print(team)
    elif counth < countd:
        print("Draw")


print(at['HomeSaves'])
print(at['AwaySaves'])

mean_acc = mean_acc/90
print(mean_acc)


