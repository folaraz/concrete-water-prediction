import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import shutil
import tensorflow as tf
import numpy as np

from model import utils
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

ROOT_DIR = '/home/folaraz/PycharmProjects/conc_proj/'

saved_model_directory = "SAVED_MODEL_LINREG"

saved_model_directory = ROOT_DIR + saved_model_directory


if os.path.isdir(saved_model_directory):
    shutil.rmtree(saved_model_directory)

FILE_NAME = '/home/folaraz/PycharmProjects/conc_proj/data/concrete.csv'
LAMBDA = 0.01
LEARNING_RATE = 0.1
N_FEATURES = 5

x_train, y_train = utils.train
x_test, y_test = utils.test


x_train = np.matrix(x_train).reshape(344, 5)
y_train = np.matrix(y_train.reshape(344,1))

x_test = np.matrix(x_test).reshape(39,5)
y_test = np.matrix(y_test.reshape(39,1))


X = tf.placeholder(tf.float32, shape=[None, N_FEATURES], name="input")

Y = tf.placeholder(tf.float32, shape=[None, 1],name='water')

W = tf.get_variable('W1', initializer=tf.random_normal(([N_FEATURES, 1])))


b = tf.get_variable('bias', initializer=tf.constant(0.1,shape=[]))

# Step 4: build model to predict Y
Y_predicted = tf.add(tf.matmul(X, W), b, name="y_predicted")


loss = tf.reduce_mean(tf.square(Y - Y_predicted), name='loss')

# Step 6: using gradient descent with learning rate of 0.001 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.3).minimize(loss)


builder = tf.saved_model.builder.SavedModelBuilder(saved_model_directory)


start = time.time()
with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    sess.run(tf.global_variables_initializer())

    # Step 8: train the model for 100 epochs
    for i in range(100):
        sess.run(optimizer, feed_dict={X: x_train, Y: y_train})

    builder.add_meta_graph_and_variables(sess, ["LINREG"])

    w_computed = sess.run(W)
    b_computed = sess.run(b)


# do a forward pass
    print("Predicted price: " + str(utils.predict.output(sess.run(Y_predicted,
                                                            feed_dict={X: utils.predict.input(314.3, 628.6, 1257.1, 28,
                                                                                        29.00)}))))
print('Took: %f seconds' % (time.time() - start))
print("w computed [%s]" % ', '.join(['%.5f' % x for x in w_computed.flatten()]))
print("b computed %.3f" % b_computed)
builder.save()

