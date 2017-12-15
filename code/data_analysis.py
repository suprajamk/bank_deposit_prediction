import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn import metrics as m
from sklearn.utils import resample
from sklearn.model_selection import train_test_split


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

test_id = test_df['id']
df = train_df

#remove age outliers
df = df[df.age < 89]
df = df[df.age > 18]

df_majority = df[df.y == 'no']
df_minority = df[df.y == 'yes']

print(df['y'].value_counts())

# Upsample minority class
df_minority_upsampled = resample(df_minority,
                                 replace=True,  # sample with replacement
                                 n_samples=20000,  # to match majority class
                                 random_state=123)


#downsample majority class
df_majority_downsampled = resample(df_majority,
                                   replace=True,  # sample with replacement
                                 n_samples=20000,  # to match minority class
                                 random_state=123)

# Combine downsampled majority class with upsampled minority class
df_sampled = pd.concat([df_majority_downsampled, df_minority_upsampled])

train_df = df_sampled
train_df_y = df_sampled['y']

#Split to train,validation
X_train, X_val, Y_train, Y_val = train_test_split(train_df, train_df_y, test_size=.1, random_state=10)


X_datas = [X_train, X_val]


def convert_categorical_to_numeric(data):
    data.job.replace(('admin.','blue-collar','management','entrepreneur','housemaid','retired','technician','services',
                      'student','unemployed','self-employed','unknown'),(0,1,2,3,4,5,6,7,8,9,10,11), inplace=True)
    data.marital.replace(('married', 'single', 'divorced','unknown'), (1, 2, 3, 0), inplace=True)
    data.education.replace(('high.school','university.degree','professional.course','unknown','basic.6y','basic.4y',
                            'basic.9y','illiterate'),(0,1,2,7,3,4,5,6),inplace=True)
    data.default.replace(('yes', 'no', 'unknown'), (1, 2, 0), inplace=True)
    data.housing.replace(('yes', 'no','unknown'), (1, 2,0), inplace=True)
    data.loan.replace(('yes', 'no','unknown'), (1, 2, 0), inplace=True)
    data.contact.replace(('telephone', 'cellular'), (1, 2), inplace=True)
    data.month.replace(('jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'),
                       (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), inplace=True)
    data.day_of_week.replace(('mon', 'tue', 'wed', 'thu', 'fri'),
                             (1, 2, 3, 4, 5), inplace=True)
    data.poutcome.replace(('failure','nonexistent','success'),(1,2,0), inplace=True)
    return data

def loan_aggregate(data):
    data['hasloan'] = 0
    data.loc[(data['housing'] == 1)| (data['loan'] == 1), 'hasloan'] = 1 # has loan
    data.loc[(data['housing'] == 2) & (data['loan'] == 2), 'hasloan'] = 2 # no loan
    data.loc[(data['housing'] == 0) & (data['loan'] == 0), 'hasloan'] = 3 # unknown
    data.loc[(data['housing'] == 1) & (data['loan'] == 0), 'hasloan'] = 4 # has housing only
    data.loc[(data['housing'] == 0) & (data['loan'] == 2), 'hasloan'] = 5 # has personal loan only
    return data

def previous_campaign_aggregate(data):
    data['prev_agg'] = 0
    data.loc[(data['pdays'] == 999) & (data['previous'] == 0) & (data['poutcome'] == 2), 'prev_agg'] = 1 #nonexistent
    data.loc[(data['pdays'] == 999) & (data['previous'] >= 1) & (data['poutcome'] == 1), 'prev_agg'] = 2 #failure
    data.loc[(data['pdays'] < 999) & (data['previous'] > 0) & (data['poutcome'] == 1), 'prev_agg'] = 3 #failure
    data.loc[(data['pdays'] < 999) & (data['previous'] > 0) & (data['poutcome'] == 0), 'prev_agg'] = 4 #success or failure
    return data

def age_aggregate(data):
    data['age_agg'] = 0
    data.loc[(data['age'] <= 22), 'age'] = 1
    data.loc[(data['age'] > 22) & (data['age'] <= 29), 'age_agg'] = 2
    data.loc[(data['age'] > 29) & (data['age'] <= 40), 'age_agg'] = 3
    data.loc[(data['age'] > 40) & (data['age'] <= 60), 'age_agg'] = 4
    data.loc[(data['age'] > 60) & (data['age'] <= 80), 'age_agg'] = 5
    data.loc[(data['age'] > 80), 'age_agg'] = 6
    return data


for data in X_datas:
    data = convert_categorical_to_numeric(data)
    data = loan_aggregate(data)
    data = previous_campaign_aggregate(data)
    data = age_aggregate(data)

X_datas[0].y.replace(('yes', 'no'), (1, 0), inplace=True)
X_datas[1].y.replace(('yes', 'no'), (1, 0), inplace=True)

classifiers = {'Logistic Regression':LogisticRegression(),
               'KNN':KNeighborsClassifier(n_neighbors = 3),
               'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
               'GNB': GaussianNB(),
               'Decision Tree':DecisionTreeClassifier(),
                'Random Forest Classifier': RandomForestClassifier(),
'Gradient Boosting Classifier':GradientBoostingClassifier(learning_rate=0.1, n_estimators=200,max_depth=6,
                                                          min_samples_split=500, min_samples_leaf=45,
                                                          subsample=0.8, random_state=5, max_features=4,
                                                          warm_start=True, verbose=True)}
data_y = pd.DataFrame(X_datas[0]['y'])

data_x = X_datas[0].drop(['y', 'id', 'default', 'housing', 'loan', 'poutcome', 'pdays', 'previous'],
                         axis=1)
print(data_x.columns)
log_cols = ["Classifier", "MCC"]

log = pd.DataFrame(columns=log_cols)

y_out = pd.DataFrame(X_datas[1]['y'])

X_test_full = X_datas[1].drop(
    ['y', 'id','default','housing','loan','poutcome', 'pdays','previous'], axis=1)

for name, classifier in classifiers.items():
    print(name)
    cls = classifier
    cls = cls.fit(data_x, data_y)

    y_pred = (cls.predict_proba(X_test_full)[:, 1] > 0.65).astype(int)

    mathews = m.matthews_corrcoef(y_out, y_pred) #using MCC as dataset is skewed towards positive data
    log_entry = pd.DataFrame([[name, mathews]], columns=log_cols)
    log = log.append(log_entry)

print(log)

log.to_csv('../output/scores.csv', index=False)