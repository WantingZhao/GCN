from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import json
import os
from utils import *
from models import GCN, MLP

# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'dblp', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 50, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')


for training_per_class in [200]:
    result_path = 'result/' + FLAGS.dataset + str(training_per_class) + '/'
    if not os.path.exists('result'):
        os.mkdir('result')
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    # Load data
    metapaths, metapaths_name, adjs, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(
        FLAGS.dataset, training_per_class)
    features = preprocess_features(features)

    all_metapaths_best = (0, 0, 0, 0)
    for metapath_id in range(len(metapaths)):
        metapath_name = metapaths_name[metapath_id]
        if not (len(metapath_name) > 0 and metapath_name[0][0] == metapath_name[-1][-1] and metapath_name[-1][-1] == 'A'):
            continue
        if not (metapath_name==['AP','PC','CP','PA']):
            continue
        best = (0, 0, 0, '')
        print("metapath=", metapaths_name[metapath_id], "-----------------------------")
        adj = adjs[metapath_id]
        # Some preprocessing
        if FLAGS.model == 'gcn':
            support = [preprocess_adj(adj)]
            num_supports = 1
            model_func = GCN
        elif FLAGS.model == 'gcn_cheby':
            support = chebyshev_polynomials(adj, FLAGS.max_degree)
            num_supports = 1 + FLAGS.max_degree
            model_func = GCN
        elif FLAGS.model == 'dense':
            support = [preprocess_adj(adj)]  # Not used
            num_supports = 1
            model_func = MLP
        else:
            raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

        # Define placeholders
        placeholders = {
            'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
            'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
            'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
            'labels_mask': tf.placeholder(tf.int32),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
        }

        # Create model
        model = model_func(placeholders, input_dim=features[2][1], logging=True)

        # Initialize session
        sess = tf.Session()

        # Init variables
        sess.run(tf.global_variables_initializer())

        cost_val = []
        # 最好的验证集acc 的情况下的：验证集acc、测试集acc、第几轮、用的metapath

        # Train model
        for epoch in range(FLAGS.epochs):

            t = time.time()
            # Construct feed dictionary
            feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Training step
            outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

            # Validation
            cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
            cost_val.append(cost)

            # Print results
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
                  "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
                  "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
            if acc > best[0]:
                test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
                print("Test set results:", "cost=", "{:.5f}".format(test_cost),
                      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
                best = (acc, test_acc, epoch, str(metapaths_name[metapath_id]))


                # tensor_dict = tf.get_default_graph().get_tensor_by_name("graphconvolution_2/full:0")
                # feed_dict_val = construct_feed_dict(features, support, y_test, test_mask, placeholders)
                # output = sess.run(tensor_dict, feed_dict=feed_dict_val)
                # print('batch_feature---------------', output)
                # output = [z for z,mask in zip(output,test_mask) if mask]
                # np.save("gcn_show.npy", output)
                output = [z for z, mask in zip(y_test, test_mask) if mask]
                np.save("gcn_show_label.npy", np.array(output).argmax(axis=1))

            if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
                print("Early stopping...")
                break

        print("Optimization Finished!")

        # # Testing
        # test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
        # print("Test set results:", "cost=", "{:.5f}".format(test_cost),
        #       "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
        result = "at_best_validate: valid accuracy=%.5f, test accuracy=%.5f, epoch=%d, metapath=%s" % (
        best[0], best[1], best[2], best[3])
        print(result)
        with open(result_path + "result_%s.txt" % str(metapaths_name[metapath_id]), 'w') as fout:
            fout.write(json.dumps(result))

        if best[1] > all_metapaths_best[1]:
            all_metapaths_best = best
    result = "best result: valid accuracy=%.5f, test accuracy=%.5f, epoch=%d, metapath=%s" % (
    all_metapaths_best[0], all_metapaths_best[1], all_metapaths_best[2], all_metapaths_best[3])
    print(result)
    with open(result_path + "best_result.txt", 'w') as fout:
        fout.write(json.dumps(result))