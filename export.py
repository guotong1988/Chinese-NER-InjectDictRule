import math
# import helper
import numpy as np
import tensorflow as tf
# from tensorflow.models.rnn import rnn, rnn_cell
# from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.python.ops import rnn
from tensorflow.contrib.session_bundle import exporter

# from tensorflow.contrib.session_bundle import exporter

max_f1 = 0
learning_rate = 0.002
dropout_rate = 0.5
# batch_size = 128
batch_size = 1
num_layers = 1
emb_dim = 200
hidden_dim = 100

tag_size = 9

num_epochs = 2
num_steps = 30

num_chars = 3544
num_classes = 11
num_intent_classes = 5


def build_sub_model(inputs, input_tag, embedding_matrix=None, is_training=False, is_crf=True, weight=False, scope=None):
    """
    num_epochs = num_epochs
    num_steps = num_steps
    num_chars = num_chars
    num_classes = num_classes
    """
    # placeholder of x, y and weight
    # inputs = tf.placeholder(tf.int32, [None, num_steps])
    inputs = tf.reshape(inputs, [-1, num_steps])
    # input_tag = tf.placeholder(tf.float32, [None, num_steps, tag_size])
    input_tag = tf.reshape(input_tag, [-1, num_steps, tag_size])
    outputs = {}

    if not scope:
        scope = tf.get_variable_scope()

    with tf.variable_scope(scope):
        # char embedding
        if embedding_matrix is None:
            embedding = tf.get_variable("emb", [num_chars, emb_dim])
        else:
            embedding = tf.Variable(embedding_matrix, trainable=True, name="emb", dtype=tf.float32)

        inputs_emb = tf.nn.embedding_lookup(embedding, inputs)
        inputs_emb = tf.concat(2, [inputs_emb, input_tag])
        # inputs_emb = tf.concat([inputs_emb, input_tag], 2)

        inputs_emb = tf.transpose(inputs_emb, [1, 0, 2])
        inputs_emb = tf.reshape(inputs_emb, [-1, emb_dim + tag_size])
        inputs_emb = tf.split(0, num_steps, inputs_emb)
        # inputs_emb = tf.split(inputs_emb, num_steps, 0)

        # lstm cell
        lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(hidden_dim)
        lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(hidden_dim)

        # dropout
        if is_training:
            lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
            lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_bw, output_keep_prob=(1 - dropout_rate))

        lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_fw] * num_layers)
        lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_bw] * num_layers)

        # get the length of each sample
        length = tf.reduce_sum(tf.sign(inputs), reduction_indices=1)
        length = tf.cast(length, tf.int32)

        # forward and backward
        lstm_outputs, _, _ = rnn.bidirectional_rnn(
            lstm_cell_fw,
            lstm_cell_bw,
            inputs_emb,
            dtype=tf.float32,
            sequence_length=length
        )

        intentinput = tf.reduce_max(lstm_outputs, 0)
        intent_w = tf.get_variable("intent_w", [hidden_dim * 2, num_intent_classes])
        intent_b = tf.get_variable("intent_b", [num_intent_classes])
        intentlogits = tf.matmul(intentinput, intent_w) + intent_b

        # softmax
        lstm_outputs = tf.reshape(tf.concat(1, lstm_outputs), [-1, hidden_dim * 2])
        softmax_w = tf.get_variable("softmax_w", [hidden_dim * 2, num_classes])
        softmax_b = tf.get_variable("softmax_b", [num_classes])
        logits = tf.matmul(lstm_outputs, softmax_w) + softmax_b

        if not is_crf:
            pass
        else:
            tags_scores = tf.reshape(logits, [batch_size, num_steps, num_classes])
            transitions = tf.get_variable("transitions", [num_classes + 1, num_classes + 1])

            dummy_val = -1000
            # class_pad = tf.Variable(dummy_val * np.ones((batch_size, num_steps, 1)), dtype=tf.float32)
            class_pad = dummy_val * tf.ones([batch_size, num_steps, 1], tf.float32)

            observations = tf.concat(2, [tags_scores, class_pad])

            # begin_vec = tf.Variable(np.array([[dummy_val] * num_classes + [0] for _ in range(batch_size)]), trainable=False, dtype=tf.float32)
            # end_vec = tf.Variable(np.array([[0] + [dummy_val] * num_classes for _ in range(batch_size)]), trainable=False, dtype=tf.float32)

            W = tf.ones([batch_size, 1], tf.float32)
            begin_vec = tf.constant([dummy_val] * num_classes + [0], dtype=tf.float32)
            begin_vec = tf.mul(W, begin_vec)

            end_vec = tf.constant([0] + [dummy_val] * num_classes, dtype=tf.float32)
            end_vec = tf.mul(W, end_vec)

            begin_vec = tf.reshape(begin_vec, [batch_size, 1, num_classes + 1])
            end_vec = tf.reshape(end_vec, [batch_size, 1, num_classes + 1])

            observations = tf.concat(1, [begin_vec, observations, end_vec])

            _, max_scores, max_scores_pre = forward(observations, transitions, length)

            ###############
            # max_scores = tf.ones([batch_size, num_steps + 1, num_classes + 1], tf.float32)
            # max_scores_pre = tf.ones([batch_size, num_steps + 1, num_classes + 1], tf.float32)

            # outputs['total_path_score'] = total_path_score

            outputs['max_scores'] = max_scores

            max_scores_pre = tf.to_float(max_scores_pre, name='ToFloat')
            outputs['max_scores_pre'] = max_scores_pre

            length = tf.to_float(length, name='ToFloat')
            outputs['length'] = length

            outputs['intent_probs'] = tf.nn.softmax(intentlogits)

        return outputs


def logsumexp(x, axis=None):
    x_max = tf.reduce_max(x, reduction_indices=axis, keep_dims=True)
    x_max_ = tf.reduce_max(x, reduction_indices=axis)
    return x_max_ + tf.log(tf.reduce_sum(tf.exp(x - x_max), reduction_indices=axis))


def forward(observations, transitions, length, is_viterbi=True, return_best_seq=True):
    length = tf.reshape(length, [batch_size])
    transitions = tf.reshape(tf.concat(0, [transitions] * batch_size), [batch_size, 12, 12])
    observations = tf.reshape(observations, [batch_size, num_steps + 2, 12, 1])
    observations = tf.transpose(observations, [1, 0, 2, 3])
    previous = observations[0, :, :, :]
    max_scores = []
    max_scores_pre = []
    alphas = [previous]
    for t in range(1, num_steps + 2):
        previous = tf.reshape(previous, [batch_size, 12, 1])
        current = tf.reshape(observations[t, :, :, :], [batch_size, 1, 12])
        alpha_t = previous + current + transitions
        if is_viterbi:
            max_scores.append(tf.reduce_max(alpha_t, reduction_indices=1))
            max_scores_pre.append(tf.argmax(alpha_t, dimension=1))
        alpha_t = tf.reshape(logsumexp(alpha_t, axis=1), [batch_size, 12, 1])
        alphas.append(alpha_t)
        previous = alpha_t

    alphas = tf.reshape(tf.concat(0, alphas), [num_steps + 2, batch_size, 12, 1])
    alphas = tf.transpose(alphas, [1, 0, 2, 3])
    alphas = tf.reshape(alphas, [batch_size * (num_steps + 2), 12, 1])

    last_alphas = tf.gather(alphas, tf.range(0, batch_size) * (num_steps + 2) + length)
    last_alphas = tf.reshape(last_alphas, [batch_size, 12, 1])

    max_scores = tf.reshape(tf.concat(0, max_scores), (num_steps + 1, batch_size, 12))
    max_scores_pre = tf.reshape(tf.concat(0, max_scores_pre), (num_steps + 1, batch_size, 12))
    max_scores = tf.transpose(max_scores, [1, 0, 2])
    max_scores_pre = tf.transpose(max_scores_pre, [1, 0, 2])

    return tf.reduce_sum(logsumexp(last_alphas, axis=1)), max_scores, max_scores_pre


def run_model():
    inputs = tf.placeholder(tf.int32, name='inputs')
    input_tag = tf.placeholder(tf.float32, name='input_tag')

    # length = tf.placeholder(tf.int32, name='length')
    # transitions = tf.placeholder(tf.float32, name='transitions')
    # observations = tf.placeholder(tf.int32, name='observations')

    outputs = build_sub_model(inputs, input_tag, scope="model")

    length = tf.identity(tf.reshape(outputs['length'], [-1]), name='length')

    max_scores = tf.identity(tf.reshape(outputs['max_scores'], [-1]), name='max_scores')

    max_scores_pre = tf.identity(tf.reshape(outputs['max_scores_pre'], [-1]), name='max_scores_pre')

    intent_probs = tf.identity(tf.reshape(outputs['intent_probs'], [-1]), name='intent_probs')

    # restore all the variabls
    saver = tf.train.Saver(sharded=True)

    print("restore variables ...")
    sess = tf.Session()

    INPUT_MODEL_DIR = "/home/maqiang/music_new_new/model_merge/"
    INPUT_MODEL_FILE = "model_merge"
    PATH_MODEL = 'model_merge/model_merge'
    # PATH_MODEL = ""
    saver.restore(sess, PATH_MODEL)

    model_exporter = exporter.Exporter(saver)
    model_exporter.init(sess.graph.as_graph_def(),
                        named_graph_signatures={
                            'inputs': exporter.generic_signature({
                                'inputs': inputs,
                                'input_tag': input_tag,
                            }),
                            'outputs': exporter.generic_signature({
                                'length': length,
                                'max_scores': max_scores,
                                'max_scores_pre': max_scores_pre,
                                'intent_probs': intent_probs,
                            })
                        })

    print("export model ...")
    OUTPUT_MODEL_DIR = "/home/maqiang/music_new_new/c_model/music_180415"
    OUTPUT_MODEL_VERSION = 0
    model_exporter.export(OUTPUT_MODEL_DIR, tf.constant(OUTPUT_MODEL_VERSION), sess)

    print("Done.")


def main(argv):
    """argv[1]: configure file name
    """
    run_model()


if __name__ == '__main__':
    tf.app.run()
