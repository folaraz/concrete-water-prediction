import tensorflow as tf
from model import utils


saved_model_directory = '/home/folaraz/PycharmProjects/conc_proj/SAVED_MODEL_LINREG'


def predict(cement, fine, coarse, age, strength, LINREG=False, NN=False):
    with tf.Session() as session:
        if NN:
            tf.saved_model.loader.load(session, ["NN"], saved_model_directory)
        else:
            tf.saved_model.loader.load(session, ["LINREG"], saved_model_directory)
        result = utils.predict.output(session.run('y_predicted:0',{'input:0':
                                                                        utils.predict.input(cement,fine,coarse,age,strength)}))

        return result[0][0]


print(predict(314.3, 628.6, 1257.1, 28, 29.00, NN=True))
