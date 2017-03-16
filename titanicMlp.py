import tensorflow as tf
import numpy as np
import pandas as pd


# Dataset
train = pd.read_csv('titanic/train.csv');
test = pd.read_csv('titanic/test.csv');

drop_cols_train = ['PassengerId','Name','Survived','Cabin','Embarked','Sex','Ticket']
drop_cols_test = ['PassengerId','Name','Cabin','Embarked','Sex','Ticket']
pre_train = train.drop(drop_cols_train,axis=1)
train_x = pd.concat([ 
                        pre_train,
                        pd.get_dummies(train['Embarked']),
                        pd.get_dummies(train['Sex'])
                    ],axis=1)
train_y = pd.DataFrame({'Survived':train['Survived']});
testPid = test['PassengerId'];
pre_test= test.drop(drop_cols_test,axis=1)
test_x = pd.concat([ 
                        pre_test,
                        pd.get_dummies(test['Embarked']),
                        pd.get_dummies(test['Sex'])
                    ],axis=1)
# replacing by medians
for col in train_x.columns:
    if np.any(pd.isnull(train_x[col])): 
        train_x[col].fillna(np.median(train_x[np.logical_not(pd.isnull(train_x[col]))][col]),inplace=True)
for col in test_x.columns:
    if np.any(pd.isnull(test_x[col])): 
        test_x[col].fillna(np.median(test_x[np.logical_not(pd.isnull(test_x[col]))][col]),inplace=True)

#transform from DataFrame to matrix
train_x = train_x.as_matrix(columns=None)
test_x = test_x.as_matrix(columns=None)
train_y = train_y.as_matrix(columns=None)

# hyper parameters
learning_rate = 0.1
n_epochs = 10000
n_features = 10
n_hidden_1 = 256
n_hidden_2 = 256
x = tf.placeholder(tf.float32,([None,10]))
y = tf.placeholder(tf.float32,([None,1]))

# variables
l1 = { 'w':tf.Variable(tf.random_normal(mean=0.0, stddev=0.25, shape=[n_features,n_hidden_1])),'b':tf.Variable(tf.zeros([n_hidden_1]))}
l2 = { 'w':tf.Variable(tf.random_normal(mean=0.0, stddev=0.25, shape=[n_hidden_1,n_hidden_2])),'b':tf.Variable(tf.zeros([n_hidden_2]))}
out= { 'w':tf.Variable(tf.random_normal(mean=0.0, stddev=0.25, shape=[n_hidden_2,1])),'b':tf.Variable(tf.zeros([1]))}

#Layers
def mlp(x,l1,l2,out):
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x,l1['w']),l1['b']))
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1,l2['w']),l2['b']))
    output = tf.matmul(layer2,out['w'])+out['b']
    return tf.nn.sigmoid(output)


yhat = mlp(x,l1,l2,out)
cost = yhat-y
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# tensorflow sessions
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    pred = []
    for i in range(n_epochs-50):
        for (X,Y) in zip(train_x,train_y):
            _,c, p = sess.run([optimizer,cost, yhat],feed_dict={x:X.reshape(1,-1),y:Y.reshape(1,-1)})
        if(i%10 == 0):
            print(c)
    for X in test_x:
        p = sess.run([yhat],feed_dict={x:X.reshape(1,-1)})
        pred.extend(p)


np.array(pred).ravel()>0.5