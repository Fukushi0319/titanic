import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn import preprocessing

# Hyper parameter
learning_rate=0.01
epochs=10
batch_size=100
train_size = 800
display_step = 1
step_size=epochs*batch_size

train_data_path="../input/train.csv"
raw_dataset=pd.read_csv(train_data_path)
features=['Sex','Age','Pclass','Fare','Cabin','Embarked']
train_data=raw_dataset[features]

# Preprocess data
labelEncoder = preprocessing.LabelEncoder()
train_data['Sex'] = labelEncoder.fit_transform(train_data['Sex'])
train_data['Cabin'] = train_data['Cabin'].fillna(train_data['Cabin'].mode()[0])
train_data['Cabin'] = labelEncoder.fit_transform(train_data['Cabin'])
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])
train_data['Embarked'] = labelEncoder.fit_transform(train_data['Embarked'])

# Normalize numeric data
train_data['Age']=(train_data['Age']-train_data['Age'].mean())/train_data['Age'].std()
train_data['Fare']=(train_data['Fare']-train_data['Fare'].mean())/train_data['Fare'].std()

input_dim=len(features)
output_dim=2

# Create features and lavels
x_np=np.array(train_data.fillna(train_data.mean()))

def create_lavels(dataset):
    lavels=np.array(dataset)
    lavels_=(lavels==0).astype(int)
    return np.array([lavels,lavels_]).T

y_np=create_lavels(raw_dataset.Survived)

[x_train, x_test] = np.vsplit(x_np, [train_size])
[y_train, y_test] = np.vsplit(y_np, [train_size])

x=tf.placeholder(tf.float32,[None,input_dim])
y=tf.placeholder(tf.float32,[None,output_dim])

# Simple liner model y=|x|*w+b
def create_model(x,weight,bias):
    return tf.matmul(x,weight)+bias

w=tf.Variable(tf.random.normal(shape=(input_dim,output_dim)))
b=tf.Variable(tf.random.normal(shape=(1,output_dim)))

z=create_model(x,w,b)
# Confirm dimention is correct
print(y.shape)
print(z.shape)

#loss=tf.nn.softmax_cross_entropy_with_logits_v2(y,z)
loss = tf.reduce_sum(tf.square(y-z))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Initialization
init = tf.initialize_all_variables()

# Train my own model
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        avg_cost = 0

        for i in range(step_size):
            ind = np.random.choice(train_size, batch_size)
            x_train_batch = x_train[ind]
            y_train_batch = y_train[ind]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, loss], feed_dict={x: x_train_batch, y: y_train_batch})
            # Compute average loss
            avg_cost += c / step_size
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("cost=", avg_cost)
    
    print("Optimization Finished!")
    # Test model
    print(tf.argmax(y, 1))
    correct_prediction = tf.equal(tf.argmax(z, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: x_test, y: y_test}))

