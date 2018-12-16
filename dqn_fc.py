import numpy as np
import random
import math
import tensorflow as tf

BATCH_SIZE = 32
LR = 0.00001
EPSILON = 0.1
GAMMA = 0.9
TARGET_REPLACE_ITER = 3000
MEMORY_CAPACITY = 10000
LEARNING_STEP_COUNTER = 0
global MEMORY_COUNTER
MEMORY_COUNTER = 0

N_ACTIONS = 5
N_SLIDING = 140
N_OTHERS = 12*10
N_STATES = N_SLIDING + N_OTHERS
MEMORY = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 3))

tf_s_sliding = tf.placeholder(tf.float32,[None,N_SLIDING])
tf_s_others = tf.placeholder(tf.float32, [None, N_OTHERS])
tf_a = tf.placeholder(tf.int32, [None, ])
tf_r = tf.placeholder(tf.float32, [None, ])
tf_s_sliding_ = tf.placeholder(tf.float32, [None, N_SLIDING])
tf_s_others_ = tf.placeholder(tf.float32, [None, N_OTHERS])
input_q_values = tf.placeholder(tf.float32, [None], name='input_q_values')

with tf.variable_scope('q'):
    b1w = 0.01*tf.Variable(tf.random_normal([N_SLIDING,256],name="b1w"))
    b1b = 0.01*tf.Variable(tf.zeros([1,256]),name="b1b")
    b1lo = tf.matmul(tf_s_sliding,b1w) + b1b
    b1o = tf.nn.relu(b1lo)
    b2w = 0.01*tf.Variable(tf.random_normal([256,256]),name="b2w")
    b2b = 0.01*tf.Variable(tf.zeros([1,256]),name="b2b")
    b2lo = tf.matmul(b1o,b2w) + b2b
    b2o = tf.nn.relu(b2lo)
    tf_s_others = tf.reshape(tf_s_others, [-1, N_OTHERS])

    input_size3 = 256 + N_OTHERS
    output_size3 = 1024
    input_size4 = 1024
    output_size4 = 512
    input_size5 = 512
    output_size5 = 128
    input_size6 = 128
    output_size6 = N_ACTIONS

    b3w = 0.01 * tf.Variable(tf.random_normal([input_size3, output_size3]), name="b3w")
    b3b = tf.Variable(tf.zeros([1, output_size3]), name="b3b")
    myreal_input = tf.concat([b2o, tf_s_others], 1)
    b3lo = tf.matmul(myreal_input, b3w) + b3b
    b3o = tf.nn.relu(b3lo)
    b4w = 0.01 * tf.Variable(tf.random_normal([input_size4, output_size4]), name="b4w")
    b4b = tf.Variable(tf.zeros([1, output_size4]), name="b4b")
    b4lo = tf.matmul(b3o, b4w) + b4b
    b4o = tf.nn.relu(b4lo)
    b5w = 0.01 * tf.Variable(tf.random_normal([input_size5, output_size5]), name="b5w")
    b5b = tf.Variable(tf.zeros([1, output_size5]), name="b5b")
    b5lo = tf.matmul(b4o, b5w) + b5b
    b5o = tf.nn.relu(b5lo)
    b6w = 0.01 * tf.Variable(tf.random_normal([input_size6, output_size6]), name="b6w")
    b6b = tf.Variable(tf.zeros([1, output_size6]), name="b6b")
    q = tf.matmul(b5o, b6w) + b6b

with tf.variable_scope('q_next'):
    b1w = 0.01 * tf.Variable(tf.random_normal([N_SLIDING, 256], name="b1w"))
    b1b = 0.01 * tf.Variable(tf.zeros([1, 256]), name="b1b")
    b1lo = tf.matmul(tf_s_sliding_, b1w) + b1b
    b1o = tf.nn.relu(b1lo)
    b2w = 0.01 * tf.Variable(tf.random_normal([256, 256]), name="b2w")
    b2b = 0.01 * tf.Variable(tf.zeros([1, 256]), name="b2b")
    b2lo = tf.matmul(b1o, b2w) + b2b
    b2o = tf.nn.relu(b2lo)
    tf_s_others_ = tf.reshape(tf_s_others_, [-1, N_OTHERS])

    input_size3 = 256 + N_OTHERS
    output_size3 = 1024
    input_size4 = 1024
    output_size4 = 512
    input_size5 = 512
    output_size5 = 128
    input_size6 = 128
    output_size6 = N_ACTIONS

    b3w = 0.01 * tf.Variable(tf.random_normal([input_size3, output_size3]), name="b3w", trainable=False)
    b3b = tf.Variable(tf.zeros([1, output_size3]), name="b3b", trainable=False)
    real_input = tf.concat([b2o, tf_s_others_], 1)
    b3lo = tf.matmul(real_input, b3w) + b3b
    b3o = tf.nn.relu(b3lo)
    b4w = 0.01 * tf.Variable(tf.random_normal([input_size4, output_size4]), name="b4w", trainable=False)
    b4b = tf.Variable(tf.zeros([1, output_size4]), name="b4b", trainable=False)
    b4lo = tf.matmul(b3o, b4w) + b4b
    b4o = tf.nn.relu(b4lo)
    b5w = 0.01 * tf.Variable(tf.random_normal([input_size5, output_size5]), name="b5w", trainable=False)
    b5b = tf.Variable(tf.zeros([1, output_size5]), name="b5b", trainable=False)
    b5lo = tf.matmul(b4o, b5w) + b5b
    b5o = tf.nn.relu(b5lo)
    b6w = 0.01 * tf.Variable(tf.random_normal([input_size6, output_size6]), name="b6w", trainable=False)
    b6b = tf.Variable(tf.zeros([1, output_size6]), name="b6b", trainable=False)
    q_next = tf.matmul(b5o, b6w) + b6b

a_indices = tf.stack([tf.range(tf.shape(tf_a)[0], dtype=tf.int32), tf_a], axis=1)
q_wrt_a = tf.gather_nd(params=q, indices=a_indices)

loss = tf.reduce_mean(tf.squared_difference(input_q_values, q_wrt_a))
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

sess = tf.Session()
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())


def choose_action(s_sliding, s_others):
    #print(s_sliding)
    s_sliding = s_sliding[np.newaxis, :]
    s_others = s_others[np.newaxis, :]
    # print(s_sliding.shape)
    # print(s_others.shape)

    if np.random.uniform() < EPSILON:
        #zhongjian = sess.run(myreal_input, feed_dict={tf_s_sliding: s_sliding, tf_s_others: s_others})
        #print(zhongjian)
        actions_value = sess.run(q, feed_dict={tf_s_sliding: s_sliding, tf_s_others: s_others})
        print(actions_value)
        action = np.argmax(actions_value)
    else:
        action = np.random.randint(0, N_ACTIONS)
    return action


def store_transition(s_sliding, s_others, a, r, s_sliding_, s_others_, done):
    global MEMORY_COUNTER
    s_sliding = np.reshape(s_sliding, (140))
    s_others = np.reshape(s_others, (N_OTHERS))
    s_sliding_ = np.reshape(s_sliding_, (140))
    s_others_ = np.reshape(s_others_, (N_OTHERS))
    transition = np.hstack((s_sliding, s_others, [a, r, done], s_sliding_, s_others_))
    index = MEMORY_COUNTER % MEMORY_CAPACITY
    MEMORY[index, :] = transition
    MEMORY_COUNTER += 1


def learn():
    global LEARNING_STEP_COUNTER
    if LEARNING_STEP_COUNTER % TARGET_REPLACE_ITER == 0:
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_next')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q')
        sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])
    LEARNING_STEP_COUNTER += 1

    sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
    b_memory = MEMORY[sample_index, :]

    b_s_sliding = b_memory[:, :N_SLIDING]
    b_s_sliding = np.reshape(b_s_sliding, [-1, 140])
    b_s_others = b_memory[:, N_SLIDING:N_STATES]
    b_a = b_memory[:, N_STATES].astype(int)
    b_r = b_memory[:, N_STATES + 1]
    b_done = b_memory[:, N_STATES + 2].astype(int)
    b_s_sliding_ = b_memory[:, -N_STATES:-N_OTHERS]
    b_s_sliding_ = np.reshape(b_s_sliding_, [-1, 140])
    b_s_others_ = b_memory[:, -N_OTHERS:]

    q_target = []
    np_q_next = q_next.eval(session=sess, feed_dict={tf_s_sliding_: b_s_sliding_, tf_s_others_: b_s_others_})

    for i in range(BATCH_SIZE):
        terminal = b_done[i]
        if terminal:
            q_target.append(b_r[i])
        else:
            q_target.append(b_r[i] + GAMMA * np.max(np_q_next[i]))

    # print("loss",sess.run(loss, {tf_s: b_s, tf_a: b_a, tf_r: b_r, tf_s_: b_s_, input_q_values: q_target}))
    sess.run(train_op,
             {tf_s_sliding: b_s_sliding, tf_s_others: b_s_others, tf_a: b_a, tf_r: b_r, tf_s_sliding_: b_s_sliding_,
              tf_s_others_: b_s_others_, input_q_values: q_target})

print('\nCollecting experience...')
