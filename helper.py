#encoding:utf-8
import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')

import re
import os
import csv
import time
import pickle
import numpy as np
import pandas as pd

def getEmbedding(infile_path="embedding"):
    char2id, id_char = loadMap("meta_data/char2id")
    row_index = 0
    set_id = set()
    with open(infile_path, "rb") as infile:
        for row in infile:
            row = row.strip()
            row_index += 1
            if row_index == 1:
                num_chars = int(row.split()[0])
                emb_dim = int(row.split()[1])
                emb_matrix = np.zeros((len(char2id.keys()), emb_dim))
                continue
            row = row.decode("utf-8")
            items = row.split()
            char = items[0]
            emb_vec = [float(val) for val in items[1:]]
            if char in char2id:
                #print char2id[char]
                set_id.add(char2id[char])
                emb_matrix[char2id[char]] = emb_vec

    #print "the size of emb_matrix:", len(emb_matrix)

    for id in id_char:
        if id not in set_id:
            #print id
            emb_matrix[id] = np.random.rand(emb_dim)

    #print "the size of emb_matrix:", len(emb_matrix)

    return emb_matrix

def nextBatch(X, y, X_tag, y_intent,start_index, batch_size=128):
    last_index = start_index + batch_size
    X_batch = list(X[start_index:min(last_index, len(X))])
    #print("X_batch",X_batch)
    y_batch = list(y[start_index:min(last_index, len(X))])
    X_tag_batch = list(X_tag[start_index:min(last_index, len(X))])
    y_intent_batch = list(y_intent[start_index:min(last_index, len(X))])
    #print("y_intent_batch",y_intent_batch)

    if last_index > len(X):
        left_size = last_index - (len(X))
        for i in range(left_size):
            index = np.random.randint(len(X))
            X_batch.append(X[index])
            y_batch.append(y[index])
            X_tag_batch.append(X_tag[index])
            y_intent_batch.append(y_intent[index])

    X_batch = np.array(X_batch)
    y_batch = np.array(y_batch)
    X_tag_batch = np.array(X_tag_batch)
    y_intent_batch = np.array(y_intent_batch)
    #print("X_batch",X_batch)
    #print("y_intent_batch",y_intent_batch)
    return X_batch, y_batch, X_tag_batch, y_intent_batch

def nextRandomBatch(X, y, X_tag, y_intent, batch_size=128):
    X_batch = []
    y_batch = []
    X_tag_batch = []
    y_intent_batch = []
    for i in range(batch_size):
        index = np.random.randint(len(X))
        X_batch.append(X[index])
        y_batch.append(y[index])
        X_tag_batch.append(X_tag[index])
        y_intent_batch.append(y_intent[index])

    X_batch = np.array(X_batch)
    y_batch = np.array(y_batch)
    X_tag_batch = np.array(X_tag_batch)
    y_intent_batch = np.array(y_intent_batch)

    return X_batch, y_batch, X_tag_batch, y_intent_batch

# use "0" to padding the sentence
def padding(sample, seq_max_len):
    for i in range(len(sample)):
        if len(sample[i]) < seq_max_len:
            sample[i] += [0 for _ in range(seq_max_len - len(sample[i]))]
    return sample




def prepare(chars, labels, seq_max_len, is_padding=True):
    X = []
    y = []
    tmp_x = []
    tmp_y = []

    for record in zip(chars, labels):
        c = record[0]
        l = record[1]
        # empty line
        if c == -1:
            if len(tmp_x) <= seq_max_len:
                X.append(tmp_x)
                y.append(tmp_y)
            tmp_x = []
            tmp_y = []
        else:
            tmp_x.append(c)
            tmp_y.append(l)
    if is_padding:
        X = np.array(padding(X, seq_max_len))
    else:
        X = np.array(X)
    y = np.array(padding(y, seq_max_len))

    return X, y

def extract_entity(sentence, labels):
    entitys = []
    B_index = 0
    E_index = 0
    if "B" in labels:
        B_index = labels.index("B")
    if "E" in labels:
        E_index = labels.index("E")
    if B_index==0 and E_index==0:
      entity = []
    else:
      entity = sentence[B_index:E_index+1]
    if len(entity)>0:
        entitys.append("".join(entity))

    X_index = 0
    Z_index = 0
    if "X" in labels:
        X_index = labels.index("X")
    if "Z" in labels:
        Z_index = labels.index("Z")
    if X_index == 0 and Z_index == 0:
        entity = []
    else:
        entity = sentence[X_index:Z_index+1]
    if len(entity) > 0:
        entitys.append("".join(entity))

    U_index = 0
    W_index = 0
    if "U" in labels:
        U_index = labels.index("U")
    if "W" in labels:
        W_index = labels.index("W")
    if U_index == 0 and W_index == 0:
        entity = []
    else:
        entity = sentence[U_index:W_index+1]
    if len(entity) > 0:
        entitys.append("".join(entity))
    """
    re_entity = re.compile(r'BM*E')
    m = re_entity.search(labels)
    while m:
        entity_labels = m.group()
        start_index = labels.find(entity_labels)
        entity = sentence[start_index:start_index + len(entity_labels)]
        labels = list(labels)
        # replace the "BM*E" with "OO*O"
        labels[start_index: start_index + len(entity_labels)] = ['O' for i in range(len(entity_labels))]
        entitys.append(entity)
        labels = ''.join(labels)
        m = re_entity.search(labels)

    re_entity = re.compile(r'XY*Z')
    m = re_entity.search(labels)
    while m:
        entity_labels = m.group()
        start_index = labels.find(entity_labels)
        entity = sentence[start_index:start_index + len(entity_labels)]
        labels = list(labels)
        # replace the "XY*Z" with "OO*O"
        labels[start_index: start_index + len(entity_labels)] = ['O' for i in range(len(entity_labels))]
        entitys.append(entity)
        labels = ''.join(labels)
        m = re_entity.search(labels)

    re_entity = re.compile(r'UV*W')
    m = re_entity.search(labels)
    while m:
        entity_labels = m.group()
        start_index = labels.find(entity_labels)
        entity = sentence[start_index:start_index + len(entity_labels)]
        labels = list(labels)
        # replace the "UV*W" with "OO*O"
        labels[start_index: start_index + len(entity_labels)] = ['O' for i in range(len(entity_labels))]
        entitys.append(entity)
        labels = ''.join(labels)
        m = re_entity.search(labels)
    """
    return entitys

def extractEntity_BME(sentence, labels):
    entitys = []
    re_entity = re.compile(r'BM*E')
    m = re_entity.search(labels)
    while m:
        entity_labels = m.group()
        start_index = labels.find(entity_labels)
        entity = sentence[start_index:start_index + len(entity_labels)]
        labels = list(labels)
        # replace the "BM*E" with "OO*O"
        labels[start_index: start_index + len(entity_labels)] = ['O' for i in range(len(entity_labels))]
        entitys.append(entity)
        labels = ''.join(labels)
        m = re_entity.search(labels)

    return entitys

def extractEntity_XYZ(sentence, labels):
    entitys = []
    re_entity = re.compile(r'XY*Z')
    m = re_entity.search(labels)
    while m:
        entity_labels = m.group()
        start_index = labels.find(entity_labels)
        entity = sentence[start_index:start_index + len(entity_labels)]
        labels = list(labels)
        # replace the "XY*Z" with "OO*O"
        labels[start_index: start_index + len(entity_labels)] = ['O' for i in range(len(entity_labels))]
        entitys.append(entity)
        labels = ''.join(labels)
        m = re_entity.search(labels)

    return entitys

def extractEntity_UVW(sentence, labels):
    entitys = []
    re_entity = re.compile(r'UV*W')
    m = re_entity.search(labels)
    while m:
        entity_labels = m.group()
        start_index = labels.find(entity_labels)
        entity = sentence[start_index:start_index + len(entity_labels)]
        labels = list(labels)
        # replace the "XY*Z" with "OO*O"
        labels[start_index: start_index + len(entity_labels)] = ['O' for i in range(len(entity_labels))]
        entitys.append(entity)
        labels = ''.join(labels)
        m = re_entity.search(labels)

    return entitys

def loadMap(token2id_filepath):
    if not os.path.isfile(token2id_filepath):
        print("file not exist, building map")
        build_map()

    token2id = {}
    id2token = {}
    with open(token2id_filepath,mode="r",encoding="utf-8") as infile:
        for row in infile:
            row = row.rstrip()
            token = row.split('\t')[0]
            token_id = int(row.split('\t')[1])
            token2id[token] = token_id
            id2token[token_id] = token
    return token2id, id2token

def saveMap(id2char, id2label):
    with open("meta_data/char2id", "w", encoding="utf-8") as outfile:
        for idx in id2char:
            outfile.write(id2char[idx] + "\t" + str(idx)  + "\n")
    with open("meta_data/label2id", "w", encoding="utf-8") as outfile:
        for idx in id2label:
            outfile.write(id2label[idx] + "\t" + str(idx) + "\n")
    print("saved map between token and id")

def build_map(train_path="train.in"):
    df_train = pd.read_csv(train_path, delimiter='\t', quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None, names=["char", "label"])
    chars = list(set(df_train["char"][df_train["char"].notnull()]))
    labels = list(set(df_train["label"][df_train["label"].notnull()]))
    char2id = dict(zip(chars, range(1, len(chars) + 1)))
    label2id = dict(zip(labels, range(1, len(labels) + 1)))
    id2char = dict(zip(range(1, len(chars) + 1), chars))
    id2label =  dict(zip(range(1, len(labels) + 1), labels))
    id2char[0] = "<PAD>"
    id2label[0] = "<PAD>"
    char2id["<PAD>"] = 0
    label2id["<PAD>"] = 0
    id2char[len(chars) + 1] = "<NEW>"
    char2id["<NEW>"] = len(chars) + 1

    saveMap(id2char, id2label)

    return char2id, id2char, label2id, id2label

def buildIntentMap(train_path="train.in"):
    df_train = pd.read_csv(train_path, delimiter='\t', quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None, names=["char", "label"])
    chars = list(set(df_train["char"][df_train["char"].notnull()]))
    labels = list(set(df_train["label"][df_train["label"].notnull()]))
    char2id = dict(zip(chars, range(1, len(chars) + 1)))
    label2id = dict(zip(labels, range(1, len(labels) + 1)))
    id2char = dict(zip(range(1, len(chars) + 1), chars))
    id2label =  dict(zip(range(1, len(labels) + 1), labels))
    id2char[0] = "<PAD>"
    id2label[0] = "<PAD>"
    char2id["<PAD>"] = 0
    label2id["<PAD>"] = 0
    id2char[len(chars) + 1] = "<NEW>"
    char2id["<NEW>"] = len(chars) + 1
    with open("meta_data/intentchar2id", "w") as outfile:
        for idx in id2char:
            outfile.write(id2char[idx] + "\t" + str(idx)  + "\n")
    with open("meta_data/intentlabel2id", "w") as outfile:
        for idx in id2label:
            outfile.write(id2label[idx] + "\t" + str(idx) + "\n")
    print("saved map between token and id")

    return char2id, id2char, label2id, id2label

def get_input_tag_x(input_tag_path, seq_max_len=30, is_padding=True):
    print('================================filepath')
    print(input_tag_path)
    B_label_float_list = []
    M_label_float_list = []
    E_label_float_list = []
    X_label_float_list = []
    Y_label_float_list = []
    Z_label_float_list = []
    U_label_float_list = []
    V_label_float_list = []
    W_label_float_list = []

    tmp_B = []
    tmp_M = []
    tmp_E = []
    tmp_X = []
    tmp_Y = []
    tmp_Z = []
    tmp_U = []
    tmp_V = []
    tmp_W = []

    with open(input_tag_path, "r",encoding="utf-8") as fpin:
        for line in fpin:
            line = line.strip()
            if len(line) == 0:
                if len(tmp_B) <= seq_max_len:
                    B_label_float_list.append(tmp_B)
                    M_label_float_list.append(tmp_M)
                    E_label_float_list.append(tmp_E)
                    X_label_float_list.append(tmp_X)
                    Y_label_float_list.append(tmp_Y)
                    Z_label_float_list.append(tmp_Z)
                    U_label_float_list.append(tmp_U)
                    V_label_float_list.append(tmp_V)
                    W_label_float_list.append(tmp_W)

                tmp_B = []
                tmp_M = []
                tmp_E = []
                tmp_X = []
                tmp_Y = []
                tmp_Z = []
                tmp_U = []
                tmp_V = []
                tmp_W = []

            else:
                words = line.split("\t")
                if len(words)<10:continue
                tmp_B.append(float(words[1]))
                tmp_M.append(float(words[2]))
                tmp_E.append(float(words[3]))
                tmp_X.append(float(words[4]))
                tmp_Y.append(float(words[5]))
                tmp_Z.append(float(words[6]))
                tmp_U.append(float(words[7]))
                tmp_V.append(float(words[8]))
                #print(words)
                tmp_W.append(float(words[9]))

    #if is_padding:
    B_np = np.array(padding(B_label_float_list, seq_max_len)).reshape(-1, seq_max_len, 1)
    M_np = np.array(padding(M_label_float_list, seq_max_len)).reshape(-1, seq_max_len, 1)
    E_np = np.array(padding(E_label_float_list, seq_max_len)).reshape(-1, seq_max_len, 1)
    X_np = np.array(padding(X_label_float_list, seq_max_len)).reshape(-1, seq_max_len, 1)
    Y_np = np.array(padding(Y_label_float_list, seq_max_len)).reshape(-1, seq_max_len, 1)
    Z_np = np.array(padding(Z_label_float_list, seq_max_len)).reshape(-1, seq_max_len, 1)
    U_np = np.array(padding(U_label_float_list, seq_max_len)).reshape(-1, seq_max_len, 1)
    V_np = np.array(padding(V_label_float_list, seq_max_len)).reshape(-1, seq_max_len, 1)
    W_np = np.array(padding(W_label_float_list, seq_max_len)).reshape(-1, seq_max_len, 1)

    tag_BME = np.concatenate((B_np, M_np, E_np, X_np, Y_np, Z_np, U_np, V_np, W_np), axis=2)

    #print tag_BME
    #print tag_BME.shape
    return tag_BME

def get_input_intent_y(input_tag_path, seq_max_len=30, is_padding=True):
    char2id, id2char, label2id, id2label = buildIntentMap(input_tag_path)
    df_train = pd.read_csv(input_tag_path, delimiter='\t', quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None, names=["char", "label"])
    df_train["label_id"] = df_train.label.map(lambda x : -1 if str(x) == str(np.nan) else label2id[x])
    intent=np.array(df_train["label_id"])
    #print(intent)
    return intent
def get_input_intent(input_tag_path):
    char2id, id2char = loadMap("meta_data/intentchar2id")
    label2id, id2label = loadMap("meta_data/intentlabel2id")
    #print("label2id",label2id)
    df_train = pd.read_csv(input_tag_path, delimiter='\t', quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None, names=["char", "label"])
    df_train["label_id"] = df_train.label.map(lambda x : -1 if str(x) == str(np.nan) else label2id[x])
    #intent=np.array(df_train["label_id"])
    intent=list(df_train["label_id"])
    #print((intent))
    return intent
def get_input_tag(input_tag_path, seq_max_len=30, is_padding=True):
    label_float_list = []
    tmp = []
    with open(input_tag_path, "r", encoding="utf-8") as fpin:
        for line in fpin:
            line = line.strip()
            if len(line) == 0:
                if len(tmp) <= seq_max_len:
                    tmp += [[0,0,0,0,0,0,0,0,0] for _ in range(seq_max_len - len(tmp))]
                    label_float_list.append(tmp)
                tmp = []
            else:
                words = line.split("\t")
                if len(words)<10:
                    print(line)
                    continue
                xx = []
                xx.append(float(words[1]))
                xx.append(float(words[2]))
                xx.append(float(words[3]))
                xx.append(float(words[4]))
                xx.append(float(words[5]))
                xx.append(float(words[6]))
                xx.append(float(words[7]))
                xx.append(float(words[8]))
                xx.append(float(words[9]))
                tmp.append(xx)

    return label_float_list


def get_train(train_path, val_path, input_tag_path, val_tag_path, input_intent_path, valid_intent_path, train_val_ratio=0.95, use_custom_val=False, seq_max_len=30):
    char2id, id2char, label2id, id2label = build_map(train_path)
    df_train = pd.read_csv(train_path, delimiter='\t', quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None, names=["char", "label"])

    # map the char and label into id
    df_train["char_id"] = df_train.char.map(lambda x : -1 if str(x) == str(np.nan) else char2id[x])
    df_train["label_id"] = df_train.label.map(lambda x : -1 if str(x) == str(np.nan) else label2id[x])
    #print(df_train["label_id"])
    # convert the data in maxtrix
    X, y = prepare(df_train["char_id"], df_train["label_id"], seq_max_len)
    #print(X,y)
    X_tag = get_input_tag_x(input_tag_path, seq_max_len)
    y_intent = get_input_intent_y(input_intent_path)
    print(X.shape)
    print(X_tag.shape)
    # shuffle the samples
    num_samples = len(X)
    indexs = np.arange(num_samples)
    np.random.shuffle(indexs)
    X = X[indexs]
    #print("X",X)
    y = y[indexs]
    X_tag = X_tag[indexs]
    y_intent = y_intent[indexs]
    #print("y_intent",y_intent)


    if val_path is None:
        # split the data into train and validation set
        X_train = X[:int(num_samples * train_val_ratio)]
        #print("X_train",X_train)
        y_train = y[:int(num_samples * train_val_ratio)]
        X_tag_train = X_tag[:int(num_samples * train_val_ratio)]
        y_intent_train = y_intent[:int(num_samples * train_val_ratio)]
        #print("y_intent_train",y_intent_train)

        X_val = X[int(num_samples * train_val_ratio):]
        #print("X_val",X_val)
        y_val = y[int(num_samples * train_val_ratio):]
        X_tag_val = X_tag[int(num_samples * train_val_ratio):]
        y_intent_val = y_intent[int(num_samples * train_val_ratio):]
        #print("y_intent_val",y_intent_val)
    else:
        X_train = X
        y_train = y
        X_tag_train = X_tag
        y_intent_train = y_intent
        X_val, y_val, X_tag_val, y_intent_val = get_test(val_path, val_tag_path, valid_intent_path, is_validation=True, seq_max_len=seq_max_len)

    print("train size: %d, validation size: %d" %(len(X_train), len(y_val)))

    return X_train, y_train, X_val, y_val, X_tag_train, X_tag_val, y_intent_train, y_intent_val

def get_test(test_path="test.in", test_tag_path="./song_test_char", test_intent_path="./test_intent", is_validation=False, seq_max_len=30):
    char2id, id2char = loadMap("meta_data/char2id")
    label2id, id2label = loadMap("meta_data/label2id")

    df_test = pd.read_csv(test_path, delimiter='\t', quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None, names=["char", "label"])

    def mapFunc(x, char2id):
        if str(x) == str(np.nan):
            return -1
        elif x not in char2id:
            return char2id["<NEW>"]
        else:
            return char2id[x]
    def labelMapFunc(x, label2id):
        if str(x) == str(np.nan):
            return -1
        elif x not in label2id:
            return label2id["<NEW>"]
        else:
            return label2id[x]

    df_test["char_id"] = df_test.char.map(lambda x:mapFunc(x, char2id))
    #df_test["label_id"] = df_test.label.map(lambda x : -1 if str(x) == str(np.nan) else label2id[x])
    df_test["label_id"] = df_test.label.map(lambda x : labelMapFunc(x, label2id))

    if is_validation:
        X_test, y_test = prepare(df_test["char_id"], df_test["label_id"], seq_max_len)
        X_test_tag = get_input_tag(test_tag_path, seq_max_len)
        y_intent_test = get_input_intent(test_intent_path)
        return X_test, y_test, X_test_tag, y_intent_test
    else:
        df_test["char"] = df_test.char.map(lambda x : -1 if str(x) == str(np.nan) else x)
        X_test, y_test = prepare(df_test["char_id"], df_test["label_id"], seq_max_len)
        X_test_str, y_test_str = prepare(df_test["char"], df_test["label"], seq_max_len, is_padding=False)
        X_test_tag = get_input_tag(test_tag_path, seq_max_len)
        y_intent_test = get_input_intent(test_intent_path)
        print("test size: %d" %(len(X_test)))
        return X_test, X_test_str, X_test_tag, y_test_str, y_intent_test, y_test

def get_transition(y_train_batch):
    transition_batch = []
    for m in range(len(y_train_batch)):
        y = [11] + list(y_train_batch[m]) + [0]
        for t in range(len(y)):
            if t + 1 == len(y):
                continue
            i = y[t]
            j = y[t + 1]
            if i == 0:
                break
            transition_batch.append(i * 12 + j)
    transition_batch = np.array(transition_batch)
    return transition_batch


if __name__ == "__main__":
    #in_tag = "./train_merge_BME"
    #get_input_tag(in_tag)
    #getTrain("./train_merge_BME","","","")
    input_intent_path="./intent"
    y_intent = get_input_intent(input_intent_path)
