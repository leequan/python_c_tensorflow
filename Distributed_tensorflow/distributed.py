# encoding:utf-8
import math
import tempfile
import time
import tensorflow as tf
from data_utility import *

flags = tf.app.flags

flags.DEFINE_string('ps_hosts', '192.168.2.243:22221', 'Comma-separated list of hostname:port pairs')
flags.DEFINE_string('worker_hosts', '192.168.2.249:22221,192.168.3.246:22221','Comma-separated list of hostname:port pairs')

# flags.DEFINE_string('ps_hosts', '127.0.0.1:22221', 'Comma-separated list of hostname:port pairs')
# flags.DEFINE_string('worker_hosts', '127.0.0.1:22222','Comma-separated list of hostname:port pairs')

flags.DEFINE_string('job_name', None, 'job name: worker or ps')
flags.DEFINE_integer('task_index', None, 'Index of task within the job')
flags.DEFINE_integer("issync", None, "是否采用分布式的同步模式，1表示同步模式，0表示异步模式")

flags.DEFINE_string('log_save_path', './nin_logs', 'Directory where to save tensorboard log')
flags.DEFINE_string('model_save_path', './model/', 'Directory where to save model weights')
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('iteration', 391, 'iteration')
flags.DEFINE_float('weight_decay', 0.0001, 'weight decay')
flags.DEFINE_float('dropout', 0.5, 'dropout')
flags.DEFINE_float('epochs', 164, 'epochs')
flags.DEFINE_float('momentum', 0.9, 'momentum')

FLAGS = flags.FLAGS


# ========================================================== #
# ├─ conv()
# ├─ activation(x)
# ├─ max_pool()
# └─ global_avg_pool()
# ========================================================== #

def conv(x, shape, use_bias=True, std=0.05):
    random_initializer = tf.random_normal_initializer(stddev=std)
    W = tf.get_variable('weights', shape=shape, initializer=random_initializer)
    b = tf.get_variable('bias', shape=[shape[3]], initializer=tf.zeros_initializer)
    x = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    if use_bias:
        x = tf.nn.bias_add(x,b)
    return x

def activation(x):
    return tf.nn.relu(x) 

def max_pool(input, k_size=3, stride=2):
    return tf.nn.max_pool(input, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1], padding='SAME')

def global_avg_pool(input, k_size=1, stride=1):
    return tf.nn.avg_pool(input, ksize=[1,k_size,k_size,1], strides=[1,stride,stride,1], padding='VALID')

def learning_rate_schedule(epoch_num):
      if epoch_num < 81:
          return 0.05
      elif epoch_num < 121:
          return 0.01
      else:
          return 0.001


def main(unused_argv):

    train_x, train_y, test_x, test_y = prepare_data()
    train_x, test_x = color_preprocessing(train_x, test_x)

    if FLAGS.job_name is None or FLAGS.job_name == '':
        raise ValueError('Must specify an explicit job_name !')
    else:
        print('job_name : %s' % FLAGS.job_name)
    if FLAGS.task_index is None or FLAGS.task_index == '':
        raise ValueError('Must specify an explicit task_index!')
    else:
        print('task_index : %d' % FLAGS.task_index)

    ps_spec = FLAGS.ps_hosts.split(',')
    worker_spec = FLAGS.worker_hosts.split(',')

    # 创建集群
    num_worker = len(worker_spec)
    cluster = tf.train.ClusterSpec({'ps': ps_spec, 'worker': worker_spec})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == 'ps':
        server.join()

    is_chief = (FLAGS.task_index == 0)

    with tf.device(tf.train.replica_device_setter(cluster=cluster)):
        global_step = tf.Variable(0, name='global_step', trainable=False)  # 创建纪录全局训练步数变量

        with tf.name_scope('input'):
            x  = tf.placeholder(tf.float32,[None, image_size, image_size, 3], name='input_x')
            y_ = tf.placeholder(tf.float32, [None, class_num], name='input_y')
        with tf.name_scope('keep_prob'):
            keep_prob = tf.placeholder(tf.float32)
        with tf.name_scope('learning_rate'):
            learning_rate = tf.placeholder(tf.float32)

        # build_network

        with tf.variable_scope('conv1'):
            output = conv(x,[5, 5, 3, 192],std=0.01)
            output = activation(output)

        with tf.variable_scope('mlp1-1'):
            output = conv(output,[1, 1, 192, 160])
            output = activation(output)

        with tf.variable_scope('mlp1-2'):
            output = conv(output,[1, 1, 160, 96])
            output = activation(output)

        with tf.name_scope('max_pool-1'):
            output  = max_pool(output, 3, 2)

        with tf.name_scope('dropout-1'):
            output = tf.nn.dropout(output,keep_prob)

        with tf.variable_scope('conv2'):
            output = conv(output,[5, 5, 96, 192])
            output = activation(output)

        with tf.variable_scope('mlp2-1'):
            output = conv(output,[1, 1, 192, 192])
            output = activation(output)

        with tf.variable_scope('mlp2-2'):
            output = conv(output,[1, 1, 192, 192])
            output = activation(output)

        with tf.name_scope('max_pool-2'):
            output  = max_pool(output, 3, 2)

        with tf.name_scope('dropout-2'):
            output = tf.nn.dropout(output,keep_prob)

        with tf.variable_scope('conv3'):
            output = conv(output,[3, 3, 192, 192])
            output = activation(output)

        with tf.variable_scope('mlp3-1'):
            output = conv(output,[1, 1, 192, 192])
            output = activation(output)

        with tf.variable_scope('mlp3-2'):
            output = conv(output,[1, 1, 192, 10])
            output = activation(output)

        with tf.name_scope('global_avg_pool'):
            output  = global_avg_pool(output, 8, 1)

        with tf.name_scope('moftmax'):
            output  = tf.reshape(output,[-1,10])

        # loss function: cross_entropy
        # weight decay: l2 * WEIGHT_DECAY
        # train_step: training operation

        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))

        with tf.name_scope('l2_loss'):
            l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

        with tf.name_scope('train_step'):
            train_step = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum,use_nesterov=True).minimize(cross_entropy + l2 * FLAGS.weight_decay,global_step=global_step)

        with tf.name_scope('prediction'):
            correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        # initial an saver to save model
        saver = tf.train.Saver()
        # 生成本地的参数初始化操作init_op
        init_op = tf.global_variables_initializer()
        train_dir = tempfile.mkdtemp()
        sv = tf.train.Supervisor(is_chief=is_chief, logdir=train_dir, init_op=init_op, recovery_wait_secs=1,
                                 global_step=global_step)

        if is_chief:
            print('Worker %d: Initailizing session...' % FLAGS.task_index)
        else:
            print('Worker %d: Waiting for session to be initaialized...' % FLAGS.task_index)
        # sess = sv.prepare_or_wait_for_session(server.target)
        with sv.managed_session(server.target) as sess:
            print('Worker %d: Session initialization  complete.' % FLAGS.task_index)

            # epoch = 164 
            # batch size = 128
            # iteration = 391
            # we should make sure [bath_size * iteration = data_set_number]

            for ep in range(1,FLAGS.epochs+1):
                lr = learning_rate_schedule(ep)
                pre_index = 0
                train_acc = 0.0
                train_loss = 0.0
                start_time = time.time()

                print("\nepoch %d/%d:" %(ep,FLAGS.epochs))

                for it in range(1,FLAGS.iteration+1):
                    if pre_index+FLAGS.batch_size < 50000:
                        batch_x = train_x[pre_index:pre_index+FLAGS.batch_size]
                        batch_y = train_y[pre_index:pre_index+FLAGS.batch_size]
                    else:
                        batch_x = train_x[pre_index:]
                        batch_y = train_y[pre_index:]

                    batch_x = data_augmentation(batch_x)

                    _, batch_loss, _ = sess.run([train_step,cross_entropy, global_step],feed_dict={x:batch_x, y_:batch_y, keep_prob: FLAGS.dropout, learning_rate: lr})
                    batch_acc = sess.run([accuracy],feed_dict={x:batch_x, y_:batch_y, keep_prob: 1.0})

                    train_loss += batch_loss
                    pre_index  += FLAGS.batch_size
                    print( 'Worker %d: traing step %d dome (global step:%d)' % (FLAGS.task_index, it, step))
                    print("iteration: %d/%d, train_loss: %.4f" %(it, FLAGS.iteration, train_loss / it, ) , end='\r')      

            sess.close()

if __name__ == '__main__':
    tf.app.run()
