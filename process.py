
def to_list(sentence):
    result = []
    for char in sentence:


def process(input_file_name):
    f = open(input_file_name,mode="r",encoding="utf-8")
    for line in f:
        splited = line.split("|")
        sentence = splited[0]
        sentence_ = to_list(sentence)
