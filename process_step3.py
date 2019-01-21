#!/usr/bin/python  
# -*- coding:utf-8 -*-  
import random

#"""汉字处理的工具:
#判断unicode是否是汉字，数字，英文，或者其他字符。
#全角符号转半角符号。"""
def is_chinese(uchar):
    #"""判断一个unicode是否是汉字"""
    if uchar >= u'u4e00' and uchar<=u'u9fa5':
        return True
    else:
        return False

def is_number(uchar):
    #"""判断一个unicode是否是数字"""
    if uchar >= u'\u0030' and uchar<=u'\u0039':
        return True
    else:
        return False

def is_alphabet(uchar):
    #"""判断一个unicode是否是英文字母"""
    if (uchar >= u'\u0041' and uchar<=u'\u005a') or (uchar >= u'\u0061' and uchar<=u'\u007a'):
        return True
    else:
        return False
def split(word):
    wlist=[]
    slist=[]
    sub=""
    flag="INIT"
    slot_flag="O"
    for w in word:
        if is_number(w):
            if flag=="NUM" or flag=="INIT":
                sub+=w
            else:
                flag="NUM"
                wlist.append(sub)
                slist.append(slot_flag)
                if slot_flag[0]=='B':
                    slot_flag="I" +slot_flag[1:]
                sub=w
            flag="NUM"
        elif is_alphabet(w):
            if flag=="ENG" or flag=="INIT":
                sub+=w
            else:
                wlist.append(sub)
                slist.append(slot_flag)
                if slot_flag[0]=='B':
                    slot_flag="I"+slot_flag[1:]
                sub=w
            flag="ENG"
        else:
            if w=='<':
                if flag!="INIT":
                    wlist.append(sub)
                    slist.append(slot_flag)
                    sub=""
                slot_flag = "B-singer"
                flag="INIT"
            elif w=='/':
                if flag!="INIT":
                    wlist.append(sub)
                    slist.append(slot_flag)
                    sub=""
                slot_flag = "B-song"
                flag="INIT"
            elif w==':':
                if flag!="INIT":
                    wlist.append(sub)
                    slist.append(slot_flag)
                    sub=""
                slot_flag = "B-style"
                flag="INIT"
            elif w=='>':
                wlist.append(sub)
                slist.append(slot_flag)
                slot_flag = "O"
                flag="INIT"
                sub=""
            elif w==' ':
                if flag!="INIT":
                    wlist.append(sub)
                    slist.append(slot_flag)
                    sub=""
                    if slot_flag[0]=='B':
                        slot_flag="I"+slot_flag[1:]
                    flag="INIT"
            else:
                if flag!="INIT":
                    wlist.append(sub)
                    slist.append(slot_flag)
                    if slot_flag[0]=='B':
                        slot_flag="I"+slot_flag[1:]
                sub=w
                flag="OTHER"
    if word[len(word)-1]!='>':
        wlist.append(sub)
        slist.append(slot_flag)
    return wlist, slist
def checkLabel(tmp):
    label = ""
    if tmp=="play":
        label = tmp
    elif tmp=="play_random":
        label = tmp
    elif tmp=="play_by_tag":
        label = tmp
    else:
        label = "other"
    return label
def processSlot(sent, singer, song , style):
    pos=sent.find(singer)
    while singer!= "" and pos!=-1:
        sent = sent[0:pos]+"<"+singer+">"+sent[pos+len(singer):]
        pos=sent.find(singer,pos+len(singer)+2)
    pos=sent.find(song)
    while song !="" and pos!=-1:
        sent = sent[0:pos]+"/"+song+">"+sent[pos+len(song):]
        pos=sent.find(song,pos+len(song)+2)
    pos=sent.find(style)
    while style !="" and pos!=-1:
        sent = sent[0:pos]+":"+style+">"+sent[pos+len(style):]
        pos=sent.find(style,pos+len(style)+2)
    vec, slot=split(sent)
    return vec, slot

def gettag(data,tag, sent):
    datatag=["" for i in range(len(sent))]
    i=0
    while i< len(sent):
        line = ""
        spaceline = ""
        for j in range(len(sent)-1,i-1,-1):
            line = line +  sent[j]
            spaceline = spaceline + " " + sent[j]
            if line in data or spaceline in data:
                for d in range(i,j+1):
                    datatag[d]=tag
                i=j
                break
        i+=1
    return datatag

def processLabel_old(sent, singer, song , style):
    singersent = sent
    songsent = sent
    stylesent = sent
    singertag = gettag(singer, "i", sent)
    add=0
    for i in range(0, len(singersent)):
        if i==0 and singertag[i-add]=="i":
            singersent = singersent[0:i]+"<"+singersent[i:]
            add+=1
        if i!=0 and singertag[i-1-add] !="i" and singertag[i-add]=="i":
            singersent = singersent[0:i]+"<"+singersent[i:]
            add+=1
        if i==len(singersent)-1 and singertag[i-add]=="i":
            singersent = singersent[0:i]+">"+singersent[i:]
            add+=1
        if i!=len(singersent)-1 and singertag[i+1-add]!="i" and singertag[i-add]=="i":
            singersent = singersent[0:i]+">"+singersent[i:]
            add+=1
    songtag = gettag(song, "o", sent)
    add=0
    for i in range(0, len(songsent)):
        if i==0 and songtag[i-add]=="o":
            songsent = songsent[0:i]+"/"+songsent[i:]
            add+=1
        if i!=0 and songtag[i-1-add] !="o" and songtag[i-add]=="o":
            songsent = songsent[0:i]+"/"+songsent[i:]
            add+=1
        if i==len(songsent)-1 and songtag[i-add]=="o":
            songsent = songsent[0:i]+">"+songsent[i:]
            add+=1
        if i!=len(songsent)-1 and songtag[i+1-add]!="o" and songtag[i-add]=="o":
            songsent = songsent[0:i]+">"+songsent[i:]
            add+=1
    styletag = gettag(style, "y", sent)
    add=0
    for i in range(0, len(stylesent)):
        if i==0 and styletag[i-add]=="y":
            stylesent = stylesent[0:i]+":"+stylesent[i:]
            add+=1
        if i!=0 and styletag[i-1-add] !="y" and styletag[i-add]=="y":
            stylesent = stylesent[0:i]+":"+stylesent[i:]
            add+=1
        if i==len(stylesent)-1 and styletag[i-add]=="y":
            stylesent = stylesent[0:i]+">"+stylesent[i:]
            add+=1
        if i!=len(stylesent)-1 and styletag[i+1-add]!="y" and styletag[i-add]=="y":
            stylesent = stylesent[0:i]+">"+stylesent[i:]
            add+=1
    vec, singerslot=split(singersent)
    vec2, songslot=split(songsent)
    vec3, styleslot=split(stylesent)
    return vec, singerslot, vec2, songslot, vec3, styleslot
def getBMEtag(data,tag, sent):
    datatag=[tag for i in range(len(sent))]
    i=0
    while i< len(sent):
        line = ""
        spaceline = ""
        for j in range(i, len(sent)):
            line = line + sent[j]
            if spaceline=="":
                spaceline =  sent[j]
            else:
                spaceline =  spaceline + " " + sent[j]
            if line in data or spaceline in data:
                for d in range(i,j+1):
                    if d==i:
                        datatag[d]="1"+datatag[d][1:]
                    elif d==j:
                        datatag[d]=datatag[d][:len(datatag[d])-1]+"1"
                    else :
                        datatag[d]=datatag[d][0:2]+"1"+datatag[d][3:]
        i+=1
    return datatag


def writeShuffle(max, fileName, input_label_name, output_intent_filename, sentoutName, labeloutName):
    whole=[]
    file = open(fileName,mode="r",encoding="utf-8")
    labelfile = open(input_label_name, mode="r", encoding="utf-8")
    sentout = open(sentoutName,mode='w',encoding="utf-8")
    intentfile = open(output_intent_filename, mode='w', encoding="utf-8")
    lexout = open(labeloutName,mode='w',encoding="utf-8")
    outline = ""
    eosline = ""
    lexline=""
    slen=0
    counts=0
    for line in file:
        subs=[]
        spaceline=""
        line=line.strip("\r\n")
        if line.startswith("EOS"):
            eosline=line
        #print(line)
        #print(" ".join(seg_list1).encode)
        if line!="":
            if not line.startswith("EOS"):
                outline+=line+"\n"
            slen+=1
        else:
            if slen<=30:
                sentout.write(outline+"\n")
                intentfile.write(eosline+"\n")
                counts+=1
            eosline=""
            outline=""
            slen=0
    outline=""
    slen=0
    print(counts)
    countc=0
    for line in labelfile:
        subs=[]
        spaceline=""
        line=line.strip("\r\n")
        #print(line)
        #print(" ".join(seg_list1).encode)
        if line!="":
            if not line.startswith("EOS"):
                outline+=line+"\n"
            slen+=1
        else:
            if slen<=30:
                lexout.write(outline+"\n")
                countc+=1
            outline=""
            slen=0
    print(countc)
    file.close()
    labelfile.close()
    intentfile.close()
    sentout.close()
    lexout.close()
    return

def readfile(fileName):
    file = open(fileName)
    data=set()
    linenum=0
    for line in file:
        line=line.strip()
        data.add(line.decode("utf-8"))
        linenum+=1
    file.close()
    return data
import os
if __name__=="__main__":
    #writeShuffle("train.revise2.txt","train.revise2.char.prep.txt","train.label","train.slot")
    maxl=0
    writeShuffle(maxl,"process_process_data/final_train_BME","processed_data/train_char_label","process_process_data/intent_train","process_process_data/train_BME","process_process_data/train_char_label")
    writeShuffle(maxl,"process_process_data/final_test_BME","processed_data/test_char_label","process_process_data/intent_test","process_process_data/test_BME","process_process_data/test_char_label")
    #writeShuffle("test20.revise3.txt",singerset,songset,styleset,"test_merge_BME.I","test_merge_char_label")
    #writeWithLabel("train.txt",singerset,songset,styleset,"train_merge_char_label")
    #print maxl
    os.remove("process_process_data/train_char_label")
    os.remove("process_process_data/test_char_label")
    os.remove("process_process_data/train_BME")
    os.remove("process_process_data/test_BME")
    print("write finish")
