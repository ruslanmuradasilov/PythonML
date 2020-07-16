import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

x = pd.read_csv("trainX.csv").drop(["Id"], axis=1)
x = np.array(x)
y = pd.read_csv("trainY.csv")["Value"]
y = np.array(y)
test_x = pd.read_csv("testX.csv").drop(["Id"], axis=1)
test_x = np.array(test_x)

trainX, testX, trainY, testY = train_test_split(x, y, random_state=42, test_size=0.2)

from sklearn.svm import SVR
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

scaler = StandardScaler().fit(trainX)
rescaled_trainX = scaler.transform(trainX)

parameters = {'C': uniform(), 'epsilon': uniform()}
model = SVR()
clf = RandomizedSearchCV(estimator=model, param_distributions=parameters, n_iter=100, random_state=7, cv=3)
clf.fit(rescaled_trainX, trainY)

print("Best score: %0.3f" % clf.best_score_)
print("Best parameters set:")
best_parameters = clf.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

clf = SVR(C=best_parameters['C'], epsilon=best_parameters['epsilon'])
clf.fit(rescaled_trainX, trainY)
rescaled_testX = scaler.transform(testX)
y_pred = clf.predict(rescaled_testX)
print(r2_score(testY, y_pred))

print(cross_val_score(clf, trainX, trainY, cv=5).mean())

rescaled_test_x = scaler.transform(test_x)
y_pred = clf.predict(rescaled_test_x)

Ypd = pd.DataFrame({'Value': y_pred})
Ypd['Id'] = range(len(Ypd))
Ypd.to_csv('Muradasilov_Ruslan.csv', index=False)

