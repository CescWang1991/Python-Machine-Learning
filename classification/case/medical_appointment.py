#! python 3
# medical_appointment.py -

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics

# read csv file into list in python
path = '../data/KaggleV2-May-2016.csv'
data = pd.read_csv(path)

# plot features as bar charts
def features_plots(vars):
    plt.figure(figsize=(15, 24.5))
    for i, dv in enumerate(vars):
        plt.subplot(7, 2, i + 3)
        data[dv].value_counts().plot(kind='bar', title=dv)
        plt.ylabel('Frequency')
    plt.show()

def model_performance(model_name, y_test, y_pred):

    print('Model name: %s' % model_name)
    print('Test accuracy (Accuracy Score): %f' % metrics.accuracy_score(y_test, y_pred))
    print('Test accuracy (ROC AUC Score): %f' % metrics.roc_auc_score(y_test, y_pred))

    fpr, tpr, thresholds = metrics.precision_recall_curve(y_test, y_pred)
    print('Area Under the Precision-Recall Curve: %f' % metrics.auc(fpr, tpr))
    print('\n')

    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

# vars = ['Gender', 'Age', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received', 'No-show']
# features_plots(vars)

data = data[data['Age'] >= 0]        # Removing Observations with Negative Age Values
del data['Handcap']         #  Removing Variable Named ‘Handicap’ from the Dataset
# vars = ['Gender', 'Age', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'SMS_received', 'No-show']
# features_plots(vars)

# Breaking Date Features(ScheduledDay, AppointmentDay) into Date Components
for col in ['ScheduledDay', 'AppointmentDay']:
    for i, component in enumerate(['year', 'month', 'day']):
        data['%s_%s' % (col, component)] = data[col].apply(lambda x: int(x.split('T')[0].split('-')[i]))
# Breaking ScheduledDay into Time Components
for j, component in enumerate(['hour', 'min', 'sec']):
    data['%s_%s' % ('Scheduled', component)] = \
    data['ScheduledDay'].apply(lambda x: int(x.split('T')[1][:-1].split(':')[j]))

# Features in Gender change to integers(F-0, M-1)
data['Gender'] = data['Gender'].apply(lambda x: int(0.0) if(x == 'F') else int(1.0))

# Features in No-show change to integers(no-0, yes-1)
data['Label'] = data['No-show'].apply(lambda x: int(0.0) if(x == 'No') else int(1.0))

selected_features = ['Gender', 'Age',
                     'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'SMS_received',
                     'ScheduledDay_year', 'ScheduledDay_month', 'ScheduledDay_day',
                     'AppointmentDay_year', 'AppointmentDay_month', 'AppointmentDay_day',
                     'Scheduled_hour', 'Scheduled_min', 'Scheduled_sec']

features = np.array(data[selected_features])
labels = np.array(data['Label'])
featTrain, featTest = np.array_split(features, [int(features.shape[0] * 0.7)])
labelTrain, labelTest = np.array_split(labels, [int(features.shape[0] * 0.7)])

# Train the model by applying decision tree
dt = DecisionTreeClassifier()
dt.fit(featTrain, labelTrain)
dtPred = dt.predict(featTest)
model_performance('Decision Tree Classifier', labelTest, dtPred)

# Train the model by applying Kernel Approximation with SGD Classify
rbf_feature = RBFSampler(gamma=1, random_state=1)       #The RBFSampler constructs an approximate mapping for the radial basis function kernel
featTrainSGD = rbf_feature.fit_transform(featTrain)
sgd = SGDClassifier(max_iter= 1000, tol= 0.001)
sgd.fit(featTrainSGD, labelTrain)
featTestSGD = rbf_feature.fit_transform(featTest)
sgdPred = sgd.predict(featTestSGD)
model_performance('SGD Classifier', labelTest, sgdPred)

# Training RandomForest Classifier on Training Dataset
rf = RandomForestClassifier()
rf.fit(featTrain, labelTrain)
rdPred = rf.predict(featTest)
model_performance('Random Forest Classifier', labelTest, rdPred)

# Training the Model by Applying Gradient Boosting Classifier
gb = GradientBoostingClassifier(random_state=10, learning_rate=0.1, n_estimators=200, max_depth=5, max_features=10)
gb.fit(featTrain, labelTrain)
gbPred = gb.predict(featTest)
model_performance('Gradient Boosting Classifier', labelTest, gbPred)

# Printing Features’ Weight as Assigned by Gradient Boosting Classifier
for feature, score in zip(selected_features, list(gb.feature_importances_)):
    print('%s\t%f'%(feature, score))
