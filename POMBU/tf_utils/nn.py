import tensorflow as tf
from tensorflow.contrib import layers

def fc_wn(x_, nh, scope, nonlinearity=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        V = tf.get_variable('V', shape=[int(x_.get_shape()[1]),nh], 
                              dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0, 0.05), 
                              trainable=True)
        g = tf.get_variable('g', shape=[nh], 
                              dtype=tf.float32,
                              initializer=tf.constant_initializer(1.), 
                              trainable=True)
        b = tf.get_variable('b', shape=[nh], 
                              dtype=tf.float32,
                              initializer=tf.constant_initializer(0.), 
                              trainable=True)

        # use weight normalization (Salimans & Kingma, 2016)
        x = tf.matmul(x_, V)
        scaler = g / tf.sqrt(tf.reduce_sum(tf.square(V), [0]))
        x = scaler * x + b

        if nonlinearity is not None:
            x = nonlinearity(x)
        return x


def fc_bn(x_, nh, scope, is_training, nonlinearity=None, reg_coef=5e-5):
    with tf.variable_scope(scope):
        h1 = layers.fully_connected(x_, nh, activation_fn=None, 
                                            biases_initializer=None,
                                            reuse=tf.AUTO_REUSE,
                                            scope='without_bn')
        h2 = layers.batch_norm(h1,  decay=1-reg_coef,
                                    scale=True,
                                    epsilon = 1e-4,
                                    activation_fn=nonlinearity,
                                    is_training=is_training,
                                    reuse=tf.AUTO_REUSE,
                                    scope='bn')
        return h2



def fc(x_, nh, scope, nonlinearity=None, reg_coef=5e-5):
    with tf.variable_scope(scope):
        x = layers.fully_connected(x_, nh, activation_fn=nonlinearity, 
                                            weights_regularizer=layers.l2_regularizer(float(reg_coef)),
                                            biases_regularizer=layers.l2_regularizer(float(reg_coef)),
                                            reuse=tf.AUTO_REUSE,
                                            scope='fc')
        return x



def mlp(n_out, hidden_layers, nonlinearities, weight_decays, norm="wb"):
    hidden_layers = hidden_layers.copy()
    hidden_layers.append(n_out)
    nonlinearities = nonlinearities.copy()
    nonlinearities.append(None)
    def training_network_fn(X):
        h = tf.layers.flatten(X)
        assert len(hidden_layers) == len(nonlinearities) and len(hidden_layers) == len(weight_decays)
        i = 0
        for nh, nonlinearity, decay in zip(hidden_layers, nonlinearities, weight_decays):  
            scope = 'mlp_fc{}'.format(i)
            i = i + 1
            if norm == "wn":
                h = fc_wn(h, nh, scope, nonlinearity)
            elif norm == "bn":
                h = fc_bn(h, nh, scope, True, nonlinearity, decay)
            else:
                h = fc(h, nh, scope, nonlinearity, decay)
        return h

    def prediction_network_fn(X):
        h = tf.layers.flatten(X)
        assert len(hidden_layers) == len(nonlinearities) and len(hidden_layers) == len(weight_decays)
        i = 0
        for nh, nonlinearity, decay in zip(hidden_layers, nonlinearities, weight_decays):  
            scope = 'mlp_fc{}'.format(i)
            i = i + 1
            if norm == "wn":
                h = fc_wn(h, nh, scope, nonlinearity)
            elif norm == "bn":
                h = fc_bn(h, nh, scope, False, nonlinearity, decay)
            else:
                h = fc(h, nh, scope, nonlinearity, decay)
        return h
    return training_network_fn, prediction_network_fn
