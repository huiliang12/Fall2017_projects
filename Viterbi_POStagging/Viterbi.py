# Name: HUI LIANG
# Unique: LIANGHUI

import argparse
import sys
import numpy as np
import copy

parser = argparse.ArgumentParser(description='input files: train file and test file')
parser.add_argument('input_files', metavar='input files', type=str, nargs=2, help = '2 files: train file and test file')

args = parser.parse_args()

training_file = args.input_files[0]
test_file = args.input_files[1]

print "training file: ", training_file
print "test file:  ", test_file

bigram_dict = {} 
tag_frequency = {}
word_frequency = {}


def line_processor(sequence):
    # new_sequence = 'BOS/BOS ' + sequence + 'EOS/EOS '
    new_sequence = 'BOS/BOS ' + sequence
    items = new_sequence.split()
    tags_list = []
    for item in items: 
        if "/" in item: 
            word, tag = item.rsplit("/", 1)
            # if tag == 'CD':
            #     # treating all numbers as the same word. 
            #     word = 'Cardinal number'
        else:
            # one word in train.large file has no tag
            word, tag = item, 'CD'
            word = 'Cardinal number'
        tags_list.append(tag)
        tags = tag.split("|")

        num_tags = len(tags)


        if num_tags == 1:
            tag = tags[0]
            if tag in tag_frequency:
                if word in tag_frequency[tag]:
                    tag_frequency[tag][word] += 1
                else: 
                    tag_frequency[tag][word] = 1
            else: 
                tag_frequency[tag] = {}
                tag_frequency[tag][word] = 1
        else: 
            for tag in tags:
                if tag in tag_frequency:
                    if word in tag_frequency[tag]:
                        tag_frequency[tag][word] += 1
                    else: 
                        tag_frequency[tag][word] = 1
                else: 
                    tag_frequency[tag] = {}
                    tag_frequency[tag][word] = 1

    bigrams = [bigram for bigram in zip(tags_list, tags_list[1:])]
    return bigrams


def bigram_count(bigrams_list):
    for bigram in bigrams_list:
        bigram1s = bigram[0].split("|")
        bigram2s = bigram[1].split("|")
        for tag1 in bigram1s:
            for tag2 in bigram2s:
                new_bigram = (tag1, tag2)
                if new_bigram in bigram_dict:
                    bigram_dict[new_bigram] += 1
                else:
                    bigram_dict[new_bigram] = 1

def file_handler(filename):
    with open(filename, 'r') as file:
        line_cnt = 0
        for line in file:
            line_cnt += 1
            temp_list = line_processor(line)
            bigram_count(temp_list)
            

file_handler(training_file)

for tag in tag_frequency: 
    for word in tag_frequency[tag]:
        if word not in word_frequency:
            word_frequency[word] = {}
            word_frequency[word][tag] = tag_frequency[tag][word]
        else: 
            word_frequency[word][tag] = tag_frequency[tag][word]


######################################################
# Add one smoothing for unseen word

with open(test_file, 'r') as file:
    unseen_words = []
    for line in file:
        items = line.split()
        for item in items: 
            word, tag = item.rsplit("/", 1)
            if word not in word_frequency:
                unseen_words.append(word)
    unseen_words_set = set(unseen_words)
file.close()


#print len(unseen_words_set), len(unseen_words)

for tag in tag_frequency:
    for word in tag_frequency[tag]:
        tag_frequency[tag][word] += 1

words = word_frequency.keys()
for word in unseen_words_set:
    words.append(word)

num_words = len(words)
words_index = [index for index, word in enumerate(words)]

for word in unseen_words:
    for tag in tag_frequency: 
        if word not in tag_frequency[tag]:
            tag_frequency[tag][word] = 1

######################################################
# Add one smoothing for unseen transition of states

num_tags = len(tag_frequency)
tags = tag_frequency.keys()

tags_index = [index for index, tag in enumerate(tags)]


## All tags in tag_frequency
all_states = bigram_dict.keys()
#print len(all_states)
#print all_states[:10]

# Create tag pairs for all possible tag combinations
tags_pairs = [(x, y) for x in tags for y in tags]
#print tags_pairs[:10]
#print len(tags_pairs)

# print the difference between the two tags
add_one = list(set(tags_pairs) - set(all_states))
#print add_one[:10]
#print len(add_one)

for each in add_one:
    if each not in bigram_dict:
        bigram_dict[each] = 1

#print len(bigram_dict)

#################################################

def init_matrix(num_words, num_tags):
    #print "words: ", num_words, "tags: ", num_tags
    # first create a matrix of counts
    initial_prob = np.zeros((num_tags,))
    trans_prob = np.zeros((num_tags, num_tags))
    obs_prob = np.zeros((num_tags, num_words))

    for T1_index in range(num_tags):
        T1 = tags[T1_index]
        for T2_index in range(num_tags):
            T2 = tags[T2_index]
            if (T1, T2) in bigram_dict:
                trans_prob[T1_index, T2_index] = bigram_dict[(T1, T2)]
            else: 
                #print (T1, T2)
                trans_prob[T1_index, T2_index] = 0

    for T_index in range(num_tags):
        T = tags[T_index]
        for word_idx in range(num_words):
            word = words[word_idx]
            if word in tag_frequency[T]:
                obs_prob[T_index, word_idx] = tag_frequency[T][word]
            else: 
                obs_prob[T_index, word_idx] = 0

    for T_index in range(num_tags):
        T = tags[T_index]
        if ('BOS', T) in bigram_dict:
            initial_prob[T_index] = bigram_dict[('BOS', T)]
        else: 
            initial_prob[T_index] = 0

    # fix the prob and reassign trans prob when a word has more than 1 tag. 
    new_trans_prob = trans_prob/trans_prob.sum(axis=1, keepdims=True)
    new_trans_prob[np.isnan(new_trans_prob)] = 0 # when sum is zero. divide by zero returns nan

    new_obs_prob = obs_prob/obs_prob.sum(axis=1, keepdims=True)
    new_obs_prob[np.isnan(new_obs_prob)] = 0 # when sum is zero. divide by zero returns nan

    return new_trans_prob, new_obs_prob, initial_prob

trans_prob, obs_prob, initial_prob = init_matrix(num_words, num_tags)

# np.savetxt("trans_prob.csv", trans_prob, delimiter=",")
# np.savetxt("obs_prob.csv", obs_prob, delimiter=",")
# np.savetxt("initial_prob.csv", initial_prob, delimiter=",")

#################################################
# Encode words or tags in indexes

def encode(temp_list, type):
    empty_list = []
    if type == "word":
        for each_word in temp_list:
            idx = words.index(each_word)
            empty_list.append(idx)
    if type == "tags":
        for each_tag in temp_list: 
            idx = tags.index(each_tag)
            empty_list.append(idx)
    empty_list = np.array(empty_list)
    return empty_list

def decode(sequence):
    pred_tags = []
    for i in sequence:
        tag = tags[i]
        pred_tags.append(tag)
    return pred_tags

#################################################
print "prediction starts"

output = []

# np.savetxt("tag_index.txt", new_list, fmt="%s")
# np.savetxt("word_index.txt", new_list2, fmt="%s")

incorrect_cnt = 0
correct_cnt = 0

with open(test_file, 'r') as file2:
    for line in file2: 
        wrong = 0
        idx_list = []
        correct_tags = []

        # new_line = 'BOS/BOS ' + line, no need in prediction
        pairs = line.split()
        for pair in pairs: 
            word, correct_tag = pair.rsplit("/", 1)
            idx_list.append(word)
            correct_tags.append(correct_tag)


        word_idx = encode(idx_list, "word")
        #print word_idx
        sent_length = len(word_idx)
        #print sent_length
        correct_idx = encode(correct_tags, "tags")
        #print correct_idx

        viterbi = np.zeros((num_tags, sent_length))
        backpt = np.zeros((num_tags, sent_length))
        #initial_prob = np.reshape(initial_prob, (-1,))

        viterbi[:, 0] = np.multiply(initial_prob, obs_prob[:, word_idx[0]])

        for t in range(1, sent_length):
            result = (trans_prob.T*viterbi[:, t-1]).T
            viterbi[:, t] = (np.reshape(obs_prob[:, word_idx[t]], (-1,)))*result.max(0)
            backpt[:, t] = result.argmax(0)

        seq = [0] * (sent_length)
        seq[-1]  = viterbi[:, -1].argmax()

        for ww in range(sent_length-2, -1, -1):
            # always record the previous one
            seq[ww] = int(backpt[int(seq[ww+1]), ww+1])

        wrong = sum(1 for i, j in zip(seq, correct_idx) if i != j)
        incorrect_cnt += wrong
        #print "incorrect_cnt: ", wrong, "at line ", count
        correct_cnt += len(correct_idx)
        #print "correct_cnt: ", correct_cnt

        # print out exact tags
        tag_decode = decode(seq)
        if len(tag_decode) != len(idx_list):
            print "predictions less than count"
        sent_output = [i+"/"+str(j) for i,j in zip(idx_list, tag_decode)]
        new_string = ""
        for element in sent_output:
            new_string += element + " "

        output.append(new_string)

file2.close()

print "Viterbi's tag_accuracy on test data: ", '{:.1%}'.format(float(correct_cnt - incorrect_cnt)/float(correct_cnt))

with open('POS.test.out', 'a') as f3:
    for output_str in output: 
        new_str = output_str + "\n"
        f3.write(new_str)

#################################################
#Calculate Baseline

print "Predicting Baseline"
import operator

wrong = 0
incorrect_cnt = 0
correct_cnt = 0

with open(test_file, 'r') as file2:
    count = 0
    for line in file2: 
        count += 1
        wrong = 0

        idx_list = []
        correct_tags = []

        pairs = line.split()
        for pair in pairs: 
            word, correct_tag = pair.rsplit("/", 1)
            idx_list.append(word)
            correct_tags.append(correct_tag)

        predictions = []
        for word in idx_list:
            idx = words.index(word)
            if word in word_frequency:
                pred = max(word_frequency[word].iteritems(), key=operator.itemgetter(1))[0]
            else: 
                pred = 'NN'
            predictions.append(pred)

        wrong = sum(1 for i, j in zip(predictions, correct_tags) if i != j)
        incorrect_cnt += wrong
        correct_cnt += len(correct_tags)

file2.close()
# print "total count: ", correct_cnt, " incorrect_cnt: ", incorrect_cnt
print "Baseline_accuracy on test data: ", '{:.1%}'.format(float(correct_cnt - incorrect_cnt)/float(correct_cnt))


