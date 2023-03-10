import numpy as np
import pandas as pd
import pickle as pk
import tensorflow as tf
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve

# Reading data with n columns
data = pd.read_csv('data.txt',header=None)

# Using first n-1 columns as features
X = data.iloc[:,:-1]
# Using last column as lable
y = data.iloc[:,-1]

# Splitting data into training(80%) and testing(20%)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)

# Initializing Naive Bayes model
nb_model = GaussianNB()
# Fitting Naive Bayes model
nb_model.fit(X_train,y_train)
# Scoring Naive Bayes model on training data
nb_score_train = nb_model.score(X_train, y_train)
# Scoring Naive Bayes model on testing data
nb_score_test = nb_model.score(X_test, y_test)
nb_f1_test = f1_score(y_test,nb_model.predict(X_test))
nb_cmat = confusion_matrix(y_test,nb_model.predict(X_test)).ravel()
# Creating result table for Naive Bayes model
nb_res = X_test.copy()
nb_res['actual'] = y_test
nb_res['pred'] = nb_model.predict(X_test)

# Creating list of epochs(iterations) to be used
epochs = [i+1 for i in range(10)]
# Creating graph data
train_scores , test_scores = validation_curve(
    LogisticRegression(),
    X_train,
    y_train,
    param_name="max_iter",
    param_range=epochs,
    scoring="accuracy",
    n_jobs=2
)
# Caculating mean of training scores
train_scores_mean = np.mean(train_scores, axis=1)
# Caculating mean of testing scores
test_scores_mean = np.mean(test_scores, axis=1)
# Plotting graph
plt.plot(epochs,train_scores_mean*100,marker='o',label='tain')
plt.plot(epochs,test_scores_mean*100,marker='o',label='validation')
plt.xlabel('epochs')
plt.ylabel('accuracy(%)')
plt.legend()
plt.savefig('graph_lr.png')
plt.clf()

# Initializing logistic regression model
lr_model = LogisticRegression(max_iter = 5)
# Training logistic regression model
lr_model.fit(X_train, y_train)
# Scoring logistic regression model on training data
lr_score_train = lr_model.score(X_train, y_train)
# Scoring logistic regression model on testing data
lr_score_test = lr_model.score(X_test, y_test)
lr_f1_test = f1_score(y_test,lr_model.predict(X_test))
lr_cmat = confusion_matrix(y_test,lr_model.predict(X_test)).ravel()
# Creating result table for logistic regression model
lr_res = X_test.copy()
lr_res['actual'] = y_test
lr_res['pred'] = lr_model.predict(X_test)

# Initializing neural network model
nn_model = tf.keras.Sequential()
nn_model.add(tf.keras.layers.Dense(50,activation='relu',input_shape=X_train.shape))
nn_model.add(tf.keras.layers.Dropout(0.2))
nn_model.add(tf.keras.layers.Dense(25,activation='relu'))
nn_model.add(tf.keras.layers.Dropout(0.2))
nn_model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
nn_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01,decay=0.0000001),loss='binary_crossentropy',metrics=[tf.keras.metrics.BinaryAccuracy()])
# Initializing scaler
scaler = StandardScaler()
# Training scaler
scaler.fit(X_train)
# Training neural network model
history = nn_model.fit(scaler.transform(X_train), y_train, batch_size=64, epochs=100, validation_split = 0.1)
print(history.history)
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('graph_nn.png')
plt.clf()

# Scoring neural network model on training data
nn_score_train = nn_model.evaluate(scaler.transform(X_train), y_train)[1]
# Scoring neural network model on testing data
nn_score_test = nn_model.evaluate(scaler.transform(X_test), y_test)[1]
nn_f1_test = f1_score(y_test,nn_model.predict(scaler.transform(X_test))>=0.5)
nn_cmat = confusion_matrix(y_test,nn_model.predict(scaler.transform(X_test))>=0.5).ravel()
# Creating result table for neural network model
nn_res = X_test.copy()
nn_res['actual'] = y_test
nn_res['pred'] = nn_model.predict(X_test)>=0.5

# Printing results
columns = ['Method','Accuracy','F1 Score','TN', 'FP', 'FN', 'TP']
rows = []
rows.append(['Naive Bayes',round(nb_score_test*100,2),round(nb_f1_test,3),*nb_cmat])
rows.append(['Logistic Regression',round(lr_score_test*100,2),round(lr_f1_test,3),*lr_cmat])
rows.append(['Neural Network',round(nn_score_test*100,2),round(nn_f1_test,3),*nn_cmat])
res = pd.DataFrame(rows,columns=columns)
print(res)

# Writing results
nb_res.to_csv('nb_result.csv',index=False)
lr_res.to_csv('lr_result.csv',index=False)
nn_res.to_csv('nn_result.csv',index=False)
res.to_csv('error_analysis.csv',index=False)
pk.dump(nb_model,open('nb_model.pkl','wb'))
pk.dump(lr_model,open('lr_model.pkl','wb'))
nn_model.save('nn_model')