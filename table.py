import numpy as np
import pandas as pd
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import csv

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

frames = [df1, df2, df3, df4, df5, df6, df7, df8, df9]

bdf = pd.concat(frames, axis=0, ignore_index=True)

bdf = bdf.fillna(0)
df = df.fillna(0)
cleandata(bdf)
cleandata(df)
bdf.drop(['Date'], 1, inplace=True)
X = bdf[['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HTR', 'HST', 'AST']].values
y = np.array(bdf['FTR'])

points = {'Swansea': 0, 'Burnley': 0, 'Crystal Palace': 0, 'West Brom': 0, 'Man United': 0, 'Man City': 0, 'Chelsea': 0,
          'Liverpool': 0, 'Hull': 0, 'Arsenal': 0, 'Leicester': 0, 'Sunderland': 0, 'Middlesbrough': 0, 'Stoke': 0,
          'Watford': 0, 'Bournemouth': 0, 'Southampton': 0, 'Everton': 0, 'Tottenham': 0, 'West Ham': 0,'Huddersfield':0,'Newcastle':0,'Brighton':0}



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6)
classifiers = {}

for k in range(1, 60):
	clf = RandomForestClassifier(max_depth=k, oob_score=True, n_estimators=k)
	clf.fit(X_train, y_train)
	classifiers[k] = clf

for index, row in df.iterrows():
    pre = row[['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HTR', 'HST', 'AST']].values
    pre = [pre]
    # pre.reshape(-1, 1)
    print(pre)
    

    counta = 0
    countd = 0
    counth = 0
    for k in range(1, 60):
	clf = classifiers[k]
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
            print(row['AwayTeam'])
            points[row['AwayTeam']] += 3
        elif counta < countd:
            print("Draw")
            points[row['AwayTeam']] += 1
            points[row['HomeTeam']] += 1
    elif counta <= counth:
        if counth > countd:
            print(row['HomeTeam'])
            points[row['HomeTeam']] += 3
        elif counth < countd:
            print("Draw")
            points[row['AwayTeam']] += 1
            points[row['HomeTeam']] += 1

    print(points)

    with open('outputr.csv', 'wb') as f:
        w = csv.DictWriter(f, points.keys())
        w.writeheader()
        w.writerow(points)
