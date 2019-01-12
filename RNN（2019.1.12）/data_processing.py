import  numpy as np

def data_processing ():
    files_content = ""
    with open("poetry.txt", 'r', encoding='utf-8') as f:
        for line in f:
            files_content += line.strip() + "]"
        words = sorted(list(files_content))
        word_dict = {}
        tmp = []
        for word in words:
            if word in word_dict:
                word_dict[word] += 1
            else:
                word_dict[word] = 1
        del word_dict["]"]
        for word in word_dict:
            if word_dict[word] <= 2:
                tmp += word
        for word in tmp:
            del word_dict[word]
        word_sorted = sorted(word_dict.items(), reverse=True, key=lambda x: x[1])
        words, _ = zip(*word_sorted)
        word2num = dict((c, i) for i, c in enumerate(words))
        num2word = dict((i, c) for i, c in enumerate(words))
        word2numF = lambda x: word2num.get(x, 0)
    return  word2numF, num2word, words, files_content
