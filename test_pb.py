import time
import helper
import argparse
import tensorflow as tf
from BILSTM_CRF_ATTN_NER import BILSTM_CRF
from tensorflow.python.tools import freeze_graph
# python test.py model test.in test.out -c char_emb -g 2

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="predict_output/model", help="the path of model file")
parser.add_argument("--test_path", default="processed3/test_BME", help="the path of test file")
parser.add_argument("--test_tag_path", default="processed3/test_char_label", help="the path of test tag file")
parser.add_argument("--test_intent_path", default="processed3/intent_test", help="the path of test intent file")
parser.add_argument("--output_path", default="test_output/output", help="the path of output file")
parser.add_argument("--char_emb", default="vectors.txt", help="the char embedding file")
args = parser.parse_args()

model_path = args.model_path
test_path = args.test_path
output_path = args.output_path
emb_path = args.char_emb

num_steps = 30 # it must consist with the train

start_time = time.time()

#test_tag_path = "./song_test_char"
test_tag_path = args.test_tag_path
test_intent_path = args.test_intent_path

print("preparing test data")
X_test, X_test_str, X_test_tag, y_test_str, y_test_intent, y_test, _ = helper.get_test(test_path=test_path, test_tag_path=test_tag_path, test_intent_path=test_intent_path, seq_max_len=num_steps)
char2id, id2char = helper.loadMap("meta_data/char2id")
label2id, id2label = helper.loadMap("meta_data/label2id")
intentlabel2id, id2intentlabel = helper.loadMap("meta_data/intentlabel2id")
num_chars = len(id2char.keys())
num_classes = len(id2label.keys())
num_intent_classes = len(id2intentlabel.keys())

print("num_chars : ", num_chars)
print("num_classes : ", num_classes)
#print "y_test_intent : ", y_test_intent
#print "X_test : ", X_test
#print "X_test_str : ", X_test_str
#print "X_test_tag : ", X_test_tag


if emb_path != None:
    embedding_matrix = helper.getEmbedding(emb_path)
else:
    embedding_matrix = None

print("building model")
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    inputs = tf.placeholder(tf.int32, name='inputs')
    input_tag = tf.placeholder(tf.float32, name='input_tag')

    initializer = tf.random_uniform_initializer(-0.1, 0.1)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        model = BILSTM_CRF(num_chars=num_chars, num_slot_class=num_classes, num_intent_classes=num_intent_classes,
                           num_steps=num_steps, embedding_matrix=embedding_matrix,
                           is_training=False,crf_flag=3,inputs=inputs,input_tag=input_tag)

    predicts = tf.identity(tf.reshape(tf.cast(model.predicts, tf.float32), [-1]), name='predicts')

    from functools import reduce
    from operator import mul

    def get_num_params():
        num_params = 0
        for variable in tf.all_variables():
            shape = variable.get_shape()
            p = reduce(mul, [dim.value for dim in shape], 1)
            print(variable.name,p)
            num_params += p
        return num_params

    get_num_params()

    print("loading model parameter")
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    print("testing")
    model.test(sess, X_test, X_test_str, X_test_tag, y_test_intent, y_test, output_path)
    checkpoint_prefix = "predict_output"
    output_graph_name = "output_graph_name"
    input_graph_name = "input_graph_name"
    # checkpoint_path = saver.save(sess, checkpoint_prefix + '_2/model',
    #                              latest_filename="latest_filename")
    # print(checkpoint_path)
    tf.train.write_graph(sess.graph, logdir=checkpoint_prefix, name=input_graph_name)
    input_graph_path = checkpoint_prefix+ "/" + input_graph_name

    output_node_names = "predicts"
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_graph_path = checkpoint_prefix + '/' + output_graph_name
    freeze_graph.freeze_graph(input_graph=input_graph_path,
                              input_saver="",
                              input_binary=False,
                              input_checkpoint=checkpoint_prefix+"/model",
                              output_node_names = output_node_names,
                              restore_op_name=restore_op_name,
                              filename_tensor_name=filename_tensor_name,
                              output_graph=output_graph_path,
                              clear_devices=True,
                              initializer_nodes="")

    end_time = time.time()
    print("time used %f(hour)" % ((end_time - start_time) / 3600))


    from tensorflow.contrib.session_bundle import exporter
    model_exporter = exporter.Exporter(saver)
    model_exporter.init(sess.graph.as_graph_def(),
                        named_graph_signatures={
                            'inputs': exporter.generic_signature({
                                'input_tag': input_tag,
                                'inputs': inputs,
                            }),
                            'outputs': exporter.generic_signature({
                                'predicts': predicts,
                            })
                        })

    print("export model ...")
    OUTPUT_MODEL_DIR = "cpp_model/cpp_model_pb"
    OUTPUT_MODEL_VERSION = 0
    model_exporter.export(OUTPUT_MODEL_DIR, tf.constant(OUTPUT_MODEL_VERSION), sess)
    print("export model Done.")