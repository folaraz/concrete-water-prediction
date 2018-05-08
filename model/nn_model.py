import tensorflow as tf
import utils
import os
import shutil
import time

ROOT_DIR = '/home/folaraz/PycharmProjects/conc_proj/'

saved_model_directory = "SAVED_MODEL_NN"

saved_model_directory = ROOT_DIR + saved_model_directory


if os.path.isdir(saved_model_directory):
    shutil.rmtree(saved_model_directory)


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'




FILE_NAME = '/home/folaraz/PycharmProjects/conc_proj/data/concrete.csv'
LAMBDA = 0.01
LEARNING_RATE = 0.01


x_train, y_train = utils.train
x_test, y_test = utils.test


with tf.name_scope('input'):
    # training data
    x = tf.placeholder("float", name="components")
    y = tf.placeholder("float", name="water")

with tf.name_scope('weights'):
    w1 = tf.get_variable(initializer=tf.random_normal([5, 5]), name="W1")
    w2 = tf.get_variable(initializer=tf.random_normal([5, 2]), name="W2")
    w3 = tf.get_variable(initializer=tf.random_normal([2, 1]), name="W3")

with tf.name_scope('biases'):
    # biases (we separate them from the weights because it is easier to do that when using TensorFlow)
    b1 = tf.get_variable(initializer=tf.random_normal([1, 5]), name="B1")
    b2 = tf.get_variable(initializer=tf.random_normal([1, 2]), name="B2")
    b3 = tf.get_variable(initializer=tf.random_normal([1, 1]), name="B3")

with tf.name_scope('layer_1'):
    # three hidden layer
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, w1), b1))

with tf.name_scope('layer_2'):
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, w2), b2))

with tf.name_scope('layer_3'):
    layer_3 = tf.nn.tanh(tf.add(tf.matmul(layer_2, w3), b3))

with tf.name_scope('regularization'):
    # L2 regularization applied on each weight
    regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3)

with tf.name_scope('loss'):
    # loss function + regularization value
    loss = tf.reduce_mean(tf.square(layer_3 - y)) + LAMBDA * regularization
    loss = tf.Print(loss, [loss], "loss")

with tf.name_scope('train'):
    # we'll use gradient descent as optimization algorithm
    train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

# launching the previously defined model begins here
init = tf.global_variables_initializer()

# we'll saved the model once the training is done
builder = tf.saved_model.builder.SavedModelBuilder(saved_model_directory)

with tf.Session() as session:
    session.run(init)

    # we'll make 5000 gradient descent iteration
    for i in range(20):
        session.run(train_op, feed_dict={x: x_train, y: y_train})

    builder.add_meta_graph_and_variables(session, ["NN"])

    # testing the network
    print("Testing data")
    print("Loss: " + str(session.run([loss], feed_dict={x: x_test, y: y_test})[0]))

    # do a forward pass
    print("Predicted price: " + str(utils.predict.output(session.run(layer_3,
                                                               feed_dict={x: utils.predict.input(314.3, 628.6, 1257.1, 28,
                                                                                           29.00)}))))

# saving the model
builder.save()


