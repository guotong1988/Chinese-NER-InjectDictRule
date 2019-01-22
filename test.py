import time
import helper
import argparse
import tensorflow as tf
from BILSTM_CRF import BILSTM_CRF

# python test.py model test.in test.out -c char_emb -g 2

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="predict_output/model", help="the path of model file")
parser.add_argument("--test_path", default="process_process_data/final_test_BME", help="the path of test file")
parser.add_argument("--test_tag_path", default="processed_data/test_char_label", help="the path of test tag file")
parser.add_argument("--test_intent_path", default="process_process_data/intent_test", help="the path of test intent file")
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
X_test, X_test_str, X_test_tag, y_test_str, y_test_intent, y_test = helper.get_test(test_path=test_path, test_tag_path=test_tag_path, test_intent_path=test_intent_path, seq_max_len=num_steps)
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
config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model = BILSTM_CRF(num_chars=num_chars, num_slot_class=num_classes, num_intent_classes=num_intent_classes, num_steps=num_steps, embedding_matrix=embedding_matrix, is_training=False)

        print("loading model parameter")
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        print("testing")
        model.test(sess, X_test, X_test_str, X_test_tag, y_test_intent, y_test, output_path)

        end_time = time.time()
        print("time used %f(hour)" % ((end_time - start_time) / 3600))

