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

 
def writeShuffle(fileName,sentoutName):
    whole=[]
    file = open(fileName,mode="r",encoding="utf-8")
    sentout = open(sentoutName,mode='w',encoding="utf-8")
    pre_word = ""
    pre_label = ""
    num = 0
    for line in file:
        subs=[]
        spaceline=""
        line=line.strip("\r\n")
        seg_list1 = line.split("\t")
        #print(line)
        #print(" ".join(seg_list1).encode)
        if len(seg_list1) >1:
            word = seg_list1[0]
            label = seg_list1[1]
        else:
            word = ""
            label = ""
        #seg = (" ".join(vec)).encode("utf-8")
        #slots = (" ".join(slot)).encode("utf-8")
        #print(line)
        outline="\n"
        if pre_label!="" and pre_label[0] == "I":
            if label=="" or label[0]!='I':
                pre_label = "E" + pre_label[1:]
        if pre_label=="B-song":
            pre_label="B"
        if pre_label=="I-song":
            pre_label="M"
        if pre_label=="E-song":
            pre_label="E"
        if pre_label=="B-singer":
            pre_label="X"
        if pre_label=="I-singer":
            pre_label="Y"
        if pre_label=="E-singer":
            pre_label="Z"
        if pre_label=="B-style":
            pre_label="U"
        if pre_label=="I-style":
            pre_label="V"
        if pre_label=="E-style":
            pre_label="W"

        if pre_label!="":
            outline = pre_word+"\t"+pre_label+"\n"
        if num!=0:
            sentout.write(outline)
        pre_word=word
        pre_label=label
        num+=1
        #print len(vec)
        #print(vec)
    #sentout.write(outline.encode("utf-8"))
    sentout.write("\n")
    file.close()
    sentout.close()
    return

def writeWithLabel(fileName,singerset, songset, styleset, slotoutName):
    whole=[]
    file = open(fileName)
    lex = open(slotoutName,'w')
    for line in file:
        subs=[]
        spaceline=""
        line=line.strip("\r\n")
        line=line.decode("utf-8")
        seg_list1 = line.split("\t")
        #print(line)
        #print(" ".join(seg_list1).encode)
        vec, singerslot, vec2, songslot, vec3, styleslot = processLabel(seg_list1[0],singerset,songset, styleset)
        #print(line)
        for i in range(0,len(vec)):
            outline=""
            if vec[i]!=vec2[i] or vec2[i]!=vec3[i]:
                print("error")
            if singerslot[i]=="B-singer":
                sis = "1 0 0"
            elif singerslot[i]=="I-singer":
                if i== len(vec)-1 or singerslot[i]!="I-singer":
                    sis = "0 0 1"
                else:
                    sis = "0 1 0"
            else:
                sis = "0 0 0"
            if songslot[i]=="B-song":
                sos = "1 0 0"
            elif songslot[i]=="I-song":
                if i== len(vec)-1 or songslot[i]!="I-song":
                    sos = "0 0 1"
                else:
                    sos = "0 1 0"
            else:
                sos = "0 0 0"
            if styleslot[i]=="B-style":
                sys = "1 0 0"
            elif styleslot[i]=="I-style":
                if i== len(vec)-1 or styleslot[i]!="I-style":
                    sys = "0 0 1"
                else:
                    sys = "0 1 0"
            else:
                sys = "0 0 0"

            outline+=vec[i]+"\t"+sos+"\t"+sis+"\t"+sys+"\n"
            lex.write(outline.encode("utf-8")+"\n")
        #print len(vec)
        #print(vec)
    file.close()
    lex.close()
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

if __name__=="__main__":
    #writeShuffle("train.revise2.txt","train.revise2.char.prep.txt","train.label","train.slot")
    #singerset = readfile("singer_set.txt")
    #songset = readfile("song_set.txt")
    #styleset = readfile("style_set.txt")
    writeShuffle("processed_data/train_BME","process_process_data/final_train_BME")
    writeShuffle("processed_data/test_BME","process_process_data/final_test_BME")
    #print maxl
   
    print("write finish")
