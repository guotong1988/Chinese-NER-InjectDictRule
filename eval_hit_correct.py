# -*- coding: UTF-8 -*-
import sys
import os
from os.path import join
import json
import string
import re

def eval(pathin1, pathout2, BME_hit, XYZ_hit, UVW_hit):
    dict_predict = {}
    with open(pathin1, "rb") as fpin:
        for line in fpin:
            line = line.strip()
            words = line.split("<@>")
            if len(words) == 4:
                sent = words[0].strip()
                entities = []

                tag = words[1].strip()
                if len(tag.strip()) != 0:
                    xxs = tag.split(" ")
                    for xx in xxs:
                        entities.append(xx)

                tag = words[2].strip()
                if len(tag.strip()) != 0:
                    xxs = tag.split(" ")
                    for xx in xxs:
                        entities.append(xx)

                tag = words[3].strip()
                if len(tag.strip()) != 0:
                    xxs = tag.split(" ")
                    for xx in xxs:
                        entities.append(xx)

                dict_predict[sent] = entities


    precision = -1.0
    recall = -1.0
    f1 = -1.0
    hit_num = 0
    pred_num = 0
    true_num = 0

    sent = ""
    label_t = ""

    with open(pathin2, mode="r",encoding="utf-8") as fpin:
        for line in fpin:
            if len(line.strip()) == 0:
                true_labels = extractEntity(sent, label_t)
                #pred_labels = dict_predict[sent]
                if sent not in dict_predict:
                    sent = ""
                    label_t = ""
                    continue

                pred_labels = dict_predict[sent]
                #pred_labels = extractEntity(sent, label_p)
                """
                for xx in true_labels:
                    print xx.encode("utf8"),
                print ""
                for xx in pred_labels:
                    print xx.encode("utf8"),
                print ""
                """
                #print true_labels.encode("utf8")
                #print pred_labels.encode("utf8")
                #print "sent, true, pred:",sent, set(true_labels),set(pred_labels)
                #print "all hit_num_sent", len(set(true_labels) & set(pred_labels))
                hit_num += len(set(true_labels) & set(pred_labels))
                pred_num += len(set(pred_labels))
                true_num += len(set(true_labels))
                #print hit_num, pred_num, true_num

                sent = ""
                label_t = ""
            else:
                line = line.strip()
                wors = line.split("\t")

                sent_char = str(wors[0])
                tag_orig = wors[1]

                sent = sent + sent_char
                label_t = label_t + tag_orig

    hit_num = BME_hit+XYZ_hit+UVW_hit
    print("hit_num", hit_num)
    print("pred_num", pred_num)
    print("true_num", true_num)

    if pred_num != 0:
        precision = 1.0 * (BME_hit+XYZ_hit+UVW_hit) / pred_num
    if true_num != 0:
        recall = 1.0 * (BME_hit+XYZ_hit+UVW_hit) / true_num
    if precision > 0 and recall > 0:
        f1 = 2.0 * (precision * recall) / (precision + recall)
    print("entity:", precision, recall, f1)


def eval_BME(pathin1, pathout2):
    dict_predict = {}
    with open(pathin1, "rb") as fpin:
        for line in fpin:
            line = line.strip()
            words = line.split("<@>")
            if len(words) == 4:
                sent = words[0].strip()
                tag = words[1].strip()
                if len(tag) == 0:
                    dict_predict[sent] = []
                else:
                    entities = []
                    xxs = tag.split(" ")
                    for xx in xxs:
                        entities.append(xx)
                    dict_predict[sent] = entities
            else:
                print("xx")

    precision = -1.0
    recall = -1.0
    f1 = -1.0
    hit_num = 0
    pred_num = 0
    true_num = 0

    sent = ""
    label_t = ""

    with open(pathin2, "rb") as fpin:
        for line in fpin:
            if len(line.strip()) == 0:
                true_labels = extractEntity_BME(sent, label_t)
                if sent not in dict_predict:
                    sent = ""
                    label_t = ""
                    continue

                pred_labels = dict_predict[sent]

                #print "sent, true, pred:",sent, set(true_labels),set(pred_labels)
                #print "song hit_num_sent", len(set(true_labels) & set(pred_labels))
                hit_num += len(set(true_labels) & set(pred_labels))
                pred_num += len(set(pred_labels))
                true_num += len(set(true_labels))

                #print hit_num, pred_num, true_num

                sent = ""
                label_t = ""
            else:
                line = line.strip()
                wors = line.split("\t")

                sent_char = str(wors[0])
                tag_orig = wors[1]

                sent = sent + sent_char
                label_t = label_t + tag_orig

    if pred_num != 0:
        precision = 1.0 * hit_num / pred_num
    if true_num != 0:
        recall = 1.0 * hit_num / true_num
    if precision > 0 and recall > 0:
        f1 = 2.0 * (precision * recall) / (precision + recall)
    print("song hit pred true:", hit_num, pred_num, true_num)
    print("song:", precision, recall, f1)
    return hit_num



def eval_XYZ(pathin1, pathout2):
    dict_predict = {}
    with open(pathin1, "rb") as fpin:
        for line in fpin:
            line = line.strip()
            words = line.split("<@>")
            if len(words) == 4:
                sent = words[0].strip()
                tag = words[2].strip()
                if len(tag) == 0:
                    dict_predict[sent] = []
                else:
                    entities = []
                    xxs = tag.split(" ")
                    for xx in xxs:
                        entities.append(xx)
                    dict_predict[sent] = entities

            elif len(words) == 1:
                sent = words[0].strip()
                dict_predict[sent] = "NULL"


    precision = -1.0
    recall = -1.0
    f1 = -1.0
    hit_num = 0
    pred_num = 0
    true_num = 0

    sent = ""
    label_t = ""

    with open(pathin2, "rb") as fpin:
        for line in fpin:
            if len(line.strip()) == 0:
                true_labels = extractEntity_XYZ(sent, label_t)

                if sent not in dict_predict:
                    sent = ""
                    label_t = ""
                    continue

                pred_labels = dict_predict[sent]
                #print "sent, true, pred:",sent, set(true_labels),set(pred_labels)
                #print "singer hit_num_sent", len(set(true_labels) & set(pred_labels))
                hit_num += len(set(true_labels) & set(pred_labels))
                pred_num += len(set(pred_labels))
                true_num += len(set(true_labels))

                sent = ""
                label_t = ""
            else:
                line = line.strip()
                wors = line.split("\t")

                sent_char = str(wors[0].encode("utf8"))
                tag_orig = wors[1]

                sent = sent + sent_char
                label_t = label_t + tag_orig

    if pred_num != 0:
        precision = 1.0 * hit_num / pred_num
    if true_num != 0:
        recall = 1.0 * hit_num / true_num
    if precision > 0 and recall > 0:
        f1 = 2.0 * (precision * recall) / (precision + recall)
    print("singer hit pred true:", hit_num, pred_num, true_num)
    print("singer:", precision, recall, f1)
    return hit_num

def eval_UVW(pathin1, pathout2):
    dict_predict = {}
    with open(pathin1, "rb") as fpin:
        for line in fpin:
            line = line.strip()
            words = line.split("<@>")
            if len(words) == 4:
                sent = words[0].strip()
                tag = words[3].strip()
                if len(tag) == 0:
                    dict_predict[sent] = []
                else:
                    entities = []
                    xxs = tag.split(" ")
                    for xx in xxs:
                        entities.append(xx)
                    dict_predict[sent] = entities

            elif len(words) == 1:
                sent = words[0].strip()
                dict_predict[sent] = "NULL"


    precision = -1.0
    recall = -1.0
    f1 = -1.0
    hit_num = 0
    pred_num = 0
    true_num = 0

    sent = ""
    label_t = ""

    with open(pathin2, "rb") as fpin:
        for line in fpin:
            if len(line.strip()) == 0:
                true_labels = extractEntity_UVW(sent, label_t)

                if sent not in dict_predict:
                    sent = ""
                    label_t = ""
                    continue

                pred_labels = dict_predict[sent]
                #print "sent, true, pred:",sent, set(true_labels),set(pred_labels)
                #print "style hit_num_sent", len(set(true_labels) & set(pred_labels))
                hit_num += len(set(true_labels) & set(pred_labels))
                pred_num += len(set(pred_labels))
                true_num += len(set(true_labels))

                sent = ""
                label_t = ""
            else:
                line = line.strip()
                wors = line.split("\t")

                sent_char = str(wors[0].encode("utf8"))
                tag_orig = wors[1]

                sent = sent + sent_char
                label_t = label_t + tag_orig

    if pred_num != 0:
        precision = 1.0 * hit_num / pred_num
    if true_num != 0:
        recall = 1.0 * hit_num / true_num
    if precision > 0 and recall > 0:
        f1 = 2.0 * (precision * recall) / (precision + recall)
    print("style hit pred true:", hit_num, pred_num, true_num)
    print("style:", precision, recall, f1)
    return hit_num


def extractEntity(sentence, labels):
    entitys = []
    re_entity = re.compile(r'BM*E*')
    m = re_entity.search(labels)
    while m:
        entity_labels = m.group()
        start_index = labels.find(entity_labels)
        entity = unicode(sentence, "utf8")[start_index: start_index+len(entity_labels)].encode("utf8")
        labels = list(labels)
        labels[start_index: start_index + len(entity_labels)] = ['O' for i in range(len(entity_labels))]
        entitys.append(entity.strip())
        labels = ''.join(labels)
        m = re_entity.search(labels)

    re_entity = re.compile(r'XY*Z*')
    m = re_entity.search(labels)
    while m:
        entity_labels = m.group()
        start_index = labels.find(entity_labels)
        entity = unicode(sentence, "utf8")[start_index: start_index+len(entity_labels)].encode("utf8")
        labels = list(labels)
        labels[start_index: start_index + len(entity_labels)] = ['O' for i in range(len(entity_labels))]
        entitys.append(entity.strip())
        labels = ''.join(labels)
        m = re_entity.search(labels)

    re_entity = re.compile(r'UV*W*')
    m = re_entity.search(labels)
    while m:
        entity_labels = m.group()
        start_index = labels.find(entity_labels)
        entity = unicode(sentence, "utf8")[start_index: start_index+len(entity_labels)].encode("utf8")
        labels = list(labels)
        labels[start_index: start_index + len(entity_labels)] = ['O' for i in range(len(entity_labels))]
        entitys.append(entity.strip())
        labels = ''.join(labels)
        m = re_entity.search(labels)
    return entitys


def extractEntity_BME(sentence, labels):
    entitys = []
    re_entity = re.compile(r'BM*E*')
    m = re_entity.search(labels)
    while m:
        entity_labels = m.group()
        start_index = labels.find(entity_labels)
        entity = unicode(sentence, "utf8")[start_index: start_index+len(entity_labels)].encode("utf8")
        labels = list(labels)
        labels[start_index: start_index + len(entity_labels)] = ['O' for i in range(len(entity_labels))]
        entitys.append(entity.strip())
        labels = ''.join(labels)
        m = re_entity.search(labels)
    return entitys


def extractEntity_XYZ(sentence, labels):
    entitys = []
    re_entity = re.compile(r'XY*Z*')
    m = re_entity.search(labels)
    while m:
        entity_labels = m.group()
        start_index = labels.find(entity_labels)
        entity = unicode(sentence, "utf8")[start_index: start_index+len(entity_labels)].encode("utf8")
        labels = list(labels)
        labels[start_index: start_index + len(entity_labels)] = ['O' for i in range(len(entity_labels))]
        entitys.append(entity.strip())
        labels = ''.join(labels)
        m = re_entity.search(labels)
    return entitys

def extractEntity_UVW(sentence, labels):
    entitys = []
    re_entity = re.compile(r'UV*W*')
    m = re_entity.search(labels)
    while m:
        entity_labels = m.group()
        start_index = labels.find(entity_labels)
        entity = unicode(sentence, "utf8")[start_index: start_index+len(entity_labels)].encode("utf8")
        labels = list(labels)
        labels[start_index: start_index + len(entity_labels)] = ['O' for i in range(len(entity_labels))]
        entitys.append(entity.strip())
        labels = ''.join(labels)
        m = re_entity.search(labels)
    return entitys


if __name__ == "__main__":
    pathin1 = "./test_merge_BME_out"
    pathin2 = "./test_merge_BME"
    BME_hit = eval_BME(pathin1, pathin2)
    XYZ_hit = eval_XYZ(pathin1, pathin2)
    UVW_hit = eval_UVW(pathin1, pathin2)
    eval(pathin1, pathin2, BME_hit, XYZ_hit, UVW_hit)




