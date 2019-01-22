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
        if w=='<':
            slot_flag = "B-singer"
            flag="INIT"
        elif w=='/':
            slot_flag = "B-song"
            flag="INIT"
        elif w==':':
            slot_flag = "B-style"
            flag="INIT"
        elif w=='>':
            slot_flag = "O"
            flag="INIT"
        else:
            wlist.append(w)
            slist.append(slot_flag)
            if slot_flag[0]=='B':
                slot_flag="I"+slot_flag[1:]
            flag="OTHER"
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
            spaceline =  line.replace('-',' ')
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


def writeShuffle(fileName,singerset, songset, styleset, sentoutName, lexoutName):
    whole=[]
    file = open(fileName,mode="r",encoding="utf-8")
    sentout = open(sentoutName,mode='w',encoding="utf-8")
    lexout = open(lexoutName,mode='w',encoding="utf-8")
    defaultlex="0\t0\t0\t0\t0\t0\t0\t0\t0"
    for line in file:
        subs=[]
        spaceline=""
        line = line.replace("\n",'')
        line = line.replace("\r",'')
        line=line.replace(' ','-')
        seg_list1 = line.split("|")

        #print(line)
        #print(" ".join(seg_list1).encode)
        label_tmp = seg_list1[1]
        label = checkLabel(label_tmp)
        style=""
        song=""

        if len(seg_list1)>=5:
            style=seg_list1[4]
        if len(seg_list1)>=4:
            song=seg_list1[3]

        vec, slot = processSlot(seg_list1[0],seg_list1[2],song, style)
        #seg = (" ".join(vec)).encode("utf-8")
        #slots = (" ".join(slot)).encode("utf-8")
        #print(line)
        outline=""
        for (s, k) in zip(vec, slot):
            outline+=s+"\t"+k+"\n"
        sentout.write(outline+"EOS"+"\t"+label+"\n\n")
        #print len(vec)
        songtag = getBMEtag(songset, "0\t0\t0",vec)
        singertag = getBMEtag(singerset, "0\t0\t0",vec)
        styletag = getBMEtag(styleset, "0\t0\t0",vec)
        lexline=""
        for (v, so, si, sy) in zip(vec, songtag, singertag, styletag):
            lexline+=v+"\t"+so+"\t"+si+"\t"+sy+"\n"
        lexout.write(lexline+"EOS"+"\t"+defaultlex+"\n\n")
        #print(vec)
    file.close()
    sentout.close()
    lexout.close()
    return

def readfile(fileName):
    file = open(fileName,mode="r",encoding="utf-8")
    data=set()
    linenum=0
    for line in file:
        line=line.strip()
        data.add(line)
        linenum+=1
    file.close()
    return data

if __name__=="__main__":
    #writeShuffle("train.revise2.txt","train.revise2.char.prep.txt","train.label","train.slot")
    singerset = readfile("raw_data/singer_set.txt")
    songset = readfile("raw_data/song_set.txt")
    styleset = readfile("raw_data/style_set.txt")
    writeShuffle("raw_data/train_pattern.txt",singerset,songset,styleset,"processed1/train_BME","processed1/train_char_label")
    writeShuffle("raw_data/test_pattern.txt",singerset,songset,styleset,"processed1/test_BME","processed1/test_char_label")
    #writeWithLabel("train.txt",singerset,songset,styleset,"train_merge_char_label")
    #print maxl
    print("write finish")
