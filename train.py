# import os
# os.environ['CUDA_DEVICE-ORDER'] = 'PCI_BUS_ID'


# os.environ['CUDA_VISIBLE_DEVICES'] = '6'


import time
import helper
import argparse
import tensorflow as tf
from BILSTM_CRF_NER_ADV import BILSTM_CRF

# python train.py train.in model -v validation.in -c char_emb -e 10 -g 2

parser = argparse.ArgumentParser()
parser.add_argument("--train_path", default="processed3/train_BME",help="slot")
parser.add_argument("--train_tag_path",default="processed3/train_char_label", help="knowledge info or entity vocab encoded info")
parser.add_argument("--train_intent_path", default="processed3/intent_train", help="intent")
parser.add_argument("--save_path", default="predict_output", help="the path of the saved model")
parser.add_argument("--val_path", help="the path of the validation file", default="processed3/test_BME")
parser.add_argument("--val_tag_path", help="the path of the valid tag file", default="processed3/test_char_label")
parser.add_argument("--val_intent_path", help="the path of the valid intent file", default="processed3/intent_test")
parser.add_argument("--epoch", help="the number of epoch", default=10, type=int)
parser.add_argument("--char_emb", help="the char embedding file", default="vectors.txt")
# parser.add_argument("--gpu", help="the id of gpu, the default is 0", default=0, type=int)

args = parser.parse_args()

train_path = args.train_path
save_path = args.save_path
val_path = args.val_path
val_tag_path = args.val_tag_path
num_epochs = args.epoch
emb_path = args.char_emb
# gpu_config = "/gpu:"+str(args.gpu)
num_steps = 30 # it must consist with the test

start_time = time.time()
print("preparing train and validation data")

pathin_tag = args.train_tag_path
pathin_intent =  args.train_intent_path
pathval_intent =  args.val_intent_path
X_train, y_train, X_val, y_val, X_tag_train, X_tag_val, y_intent_train,\
   y_intent_val, seq_len_train, seq_len_valid = helper.get_train(train_path=train_path, val_path=val_path, input_tag_path=pathin_tag, val_tag_path=val_tag_path, input_intent_path=pathin_intent, valid_intent_path=pathval_intent, seq_max_len=num_steps)

char2id, id2char = helper.loadMap("meta_data/char2id")
label2id, id2label = helper.loadMap("meta_data/label2id")
intentlabel2id, id2intentlabel = helper.loadMap("meta_data/intentlabel2id")
num_chars = len(id2char.keys())
num_slot_class = len(id2label.keys())
num_intent_classes = len(id2intentlabel.keys())
if emb_path != None:
    embedding_matrix = helper.getEmbedding(emb_path)
else:
    embedding_matrix = None

#pathin_tag = "./song_train_char.100"
#input_tag = get_input_tag(in_tag)

print(save_path)


def optimistic_restore(session, save_file):
    """
    restore only those variable that exists in the model
    :param session:
    :param save_file:
    :return:
    """
    reader = tf.train.NewCheckpointReader(save_file)
    # reader.get_tensor()
    saved_shapes = reader.get_variable_to_shape_map()
#    print(saved_shapes)
#    print()
#    print([var.name for var in tf.trainable_variables()])
    all_var_in_code = tf.trainable_variables()
    all_varname_in_code = []
    var_map_in_code = {}

    for var in all_var_in_code:
        all_varname_in_code.append(var.name.split(":")[0])
        var_map_in_code[var.name.split(":")[0]]=var

    restore_vars = {}
    for var_in_file in saved_shapes:
        if var_in_file in all_varname_in_code:
            restore_vars[var_in_file] = var_map_in_code[var_in_file]
    # restore_vars = { ("lm/" + v.name[6:].split(':')[0]): v for v in tf.trainable_variables() if 'bert' in v.name}
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)

print("building model")
import os
# config = tf.ConfigProto(allow_soft_placement=True)
tf.reset_default_graph()
with tf.Graph().as_default():
    inputs = tf.placeholder(tf.int32, name='inputs')
    input_tag = tf.placeholder(tf.float32, name='input_tag')

    initializer = tf.random_uniform_initializer(-0.1, 0.1)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        model = BILSTM_CRF(num_chars=num_chars, num_slot_class=num_slot_class, num_intent_classes=num_intent_classes,
                           num_steps=num_steps, num_epochs=num_epochs, embedding_matrix=embedding_matrix, crf_flag=3,
                           is_training=True,inputs=inputs,input_tag=input_tag)

    with tf.variable_scope("model", reuse=tf.AUTO_REUSE, initializer=initializer):
        model_dev = BILSTM_CRF(num_chars=num_chars, num_slot_class=num_slot_class, num_intent_classes=num_intent_classes,
                               num_steps=num_steps, num_epochs=num_epochs, embedding_matrix=embedding_matrix, crf_flag=3,
                               is_training=False,inputs=inputs,input_tag=input_tag)

    from functools import reduce
    from operator import mul


    def get_num_params():
        num_params = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            p = reduce(mul, [dim.value for dim in shape], 1)
            print(variable.name, p)
            num_params += p
        return num_params


    get_num_params()

    saver = tf.train.Saver()
    print("training model")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver.save(sess, "predict_output/model")

    if os.path.exists("predict_output"):
        # saver.restore(sess, "predict_output/model")
        optimistic_restore(sess, "predict_output/model")
    model.train(sess, saver, save_path, X_train, y_train,
                X_val, y_val, X_tag_train, X_tag_val, y_intent_train, y_intent_val,
                model_dev, seq_len_train, seq_len_valid)

    end_time = time.time()
    print("time used %f(hour)" % ((end_time - start_time) / 3600))


