import sys
import os
import re
import random
import math
import optparse
from optparse import OptionParser


import string
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

for i in string.punctuation:
    stop_words.append(i)


parser = OptionParser()
parser.add_option('-a', '--alpha', dest='alpha', type='float', action='store', default='1')
parser.add_option('-s', '--smooth', dest='using_smooth', action='store_true', default=False)
parser.add_option('-m', '--using_from', dest='using_from', action='store_true', default=False)
parser.add_option('-w', '--weight', dest='weight', type='float', action='store', default='1')
parser.add_option('-p', '--probability', dest='probability', type='float', action='store', default='1e-2')
(options, args) = parser.parse_args()

alpha = options.alpha
using_smooth = options.using_smooth
using_from = options.using_from
weight = options.weight
sample_times = 5


def smooth(times, M, y):
    if using_smooth:
        return (times + alpha) / (y + alpha * M)
    else:
        if times == 0:
            return options.probability
        else:
            return times / y


file2label = {}


term = {}
send = {}

random.seed(5)

with open('index') as f:
    for line in f.readlines():
        tmp = line.split(' ')
        file2label[tmp[1].strip('\n')] = tmp[0]

mail_pattern = 'From: .*\\w*@(\w+\\.\w*)'


def process_file(file, training):
    text = ''
    text_start = False
    sender = ''
    try:
        with open(file, 'r') as f:
            for i in f.readlines():
                if i[:4] == 'From':
                    sender = re.search(mail_pattern, i).group(1)
                if text_start:
                    text += i
                if text_start is False and i == '\n':
                    text_start = True

    except:
        return None, None
    isSpam = file2label[file]
    words = text.split()
    from_splits = sender.split('.')
    a = {word for word in words if word not in stop_words}
    b = {word for word in from_splits}

    if not training:
        return a, b
    for word in a:
        if word not in term:
            if isSpam == 'spam':
                term[word] = [1, 0]
            else:
                term[word] = [0, 1]
        else:
            if isSpam == 'spam':
                term[word][0] += 1
            else:
                term[word][1] += 1
    if using_from:
        for ele in b:
            if ele not in send:
                if isSpam == 'spam':
                    send[ele] = [1, 0]
                else:
                    send[ele] = [0, 1]
            else:
                if isSpam == 'spam':
                    send[ele][0] += 1
                else:
                    send[ele][1] += 1


def train(traing_set):
    spamNum = 0
    hamNum = 0
    for i in traing_set:
        if file2label[i] == 'spam':
            spamNum += 1
        elif file2label[i] == 'ham':
            hamNum += 1

    for file in traing_set:
        process_file(file, True)
    prob = {}
    for i in term:
        spamProb = smooth(term[i][0], len(term), spamNum)
        hamProb = smooth(term[i][1], len(term), hamNum)
        prob[i] = (math.log10(spamProb), math.log10(hamProb))
    prob['isaspam'] = math.log10(spamNum / spamNum + hamNum)
    prob['isaham'] = math.log10(hamNum / spamNum + hamNum)
    prob_send = {}
    if using_from:
        for i in send:
            spamProb = smooth(send[i][0], len(send), spamNum)
            hamProb = smooth(send[i][1], len(send), hamNum)
            prob_send[i] = (math.log10(spamProb * weight), math.log10(hamProb * weight))
    return prob, prob_send, spamNum, hamNum


def test(test_set, prob, prob_send, spam, ham):
    spamCorrect = 0.
    spamFalse = 0.
    hamCorrect = 0.
    hamFalse = 0.
    cannotopen = 0.
    for file in test_set:
        word_bag, from_bag = process_file(file, False)
        if word_bag is None:
            cannotopen += 1
            continue
        spamProb = 0
        hamProb = 0
        for i in word_bag:
            if i not in prob:
                spamProb += smooth(0, len(term), spam)
                hamProb += smooth(0, len(term), ham)
                continue
            spamProb += prob[i][0]
            hamProb += prob[i][1]
        spamProb += prob['isaspam']
        hamProb += prob['isaham']
        if using_from:
            for i in from_bag:
                if i not in prob_send:
                    spamProb += smooth(0, len(send), spam)
                    hamProb += smooth(0, len(send), ham)
                    continue
                spamProb += prob_send[i][0]
                hamProb += prob_send[i][1]
        if spamProb > hamProb:
            if file2label[file] == 'spam':
                spamCorrect += 1
            else:
                spamFalse += 1
        else:
            if file2label[file] == 'ham':
                hamCorrect += 1
            else:
                hamFalse += 1
    acc = (spamCorrect + hamCorrect) / (len(test_set) - cannotopen)
    precision = spamCorrect / (spamCorrect + spamFalse)
    recall = spamCorrect / (spamCorrect + hamFalse)
    return acc, precision, recall


percents = [0.05, 0.5, 0.75, 1]
for percent in percents:
    print('-------------------------------------------------')
    length = len(file2label)
    files = [i for i in file2label.keys()]
    cor_list = []
    min_cor = 1
    max_cor = 0
    sum_cor = 0
    min_pre = 1
    max_pre = 0
    sum_pre = 0
    min_rec = 1
    max_rec = 0
    sum_rec = 0
    min_f1 = 1
    max_f1 = 0
    sum_f1 = 0
    for i in range(0, 5):
        test_start = int(i * length / 5)
        test_end = int((i + 1) * length / 5)
        test_set = files[test_start: test_end]
        training_set = files[0: test_start] + files[test_end:]
        for m in range(0, sample_times):
            random.shuffle(training_set)
            sample_set = training_set[0: int(percent * len(training_set))]
            prob, prob_send, spam, ham = train(sample_set)
            correct_rate, precision, recall = test(test_set, prob, prob_send, spam, ham)
            f1_score = 2 * precision * recall / (precision + recall)
            min_cor = min(correct_rate, min_cor)
            max_cor = max(correct_rate, max_cor)
            sum_cor += correct_rate
            min_pre = min(precision, min_cor)
            max_pre = max(precision, max_pre)
            sum_pre += precision
            min_rec = min(recall, min_rec)
            max_rec = max(recall, max_rec)
            sum_rec += recall
            min_f1 = min(f1_score, min_f1)
            max_f1 = max(f1_score, max_f1)
            sum_f1 += f1_score
    avg_cor = round(sum_cor / 25, 4)
    avg_pre = round(sum_pre / 25, 4)
    avg_rec = round(sum_rec / 25, 4)
    avg_f1 = round(sum_f1 / 25, 4)
    min_cor = round(min_cor, 4)
    max_cor = round(max_cor, 4)
    min_pre = round(min_pre, 4)
    max_pre = round(max_pre, 4)
    min_rec = round(min_rec, 4)
    max_rec = round(max_rec, 4)
    min_f1 = round(min_f1, 4)
    max_f1 = round(max_f1, 4)
    print('sampled {} of traing set'.format(percent))
    print('''
 |   |accuracy |precision|recall    |f1 score|
 |---|---------|---------|----------|--------|
 |min|{}|{}|{}|{}|
 |max|{}|{}|{}|{}|
 |avg|{}|{}|{}|{}|'''.format(min_cor, min_pre, min_rec, min_f1, max_cor, max_pre, max_rec, max_f1, avg_cor, avg_pre, avg_rec, avg_f1))