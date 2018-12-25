import math;
import numpy;

def add_to_bigram_dict(bigram, prev_word, word):
    if prev_word in bigram:
        bigram[prev_word]['count'] = 1 + bigram[prev_word]['count']
        bigram[prev_word]['successors'][word] = 1 + bigram[prev_word]['successors'].get(word, 0)
    else:
        bigram[prev_word] = {'count': 1, 'successors': {word: 1}}


def inc_dict_value(current_dict, key):
    current_dict[key] = 1 + current_dict.get(key, 0)


def process_corpus(file_name):
    unigram = {}
    tag_bigram = {}
    word_tag_bigram = {}
    word_count = {}
    N = 0
    start_token='<s>'
    end_token='</s>'
    line_number = -1
    word_set = set()
    with open(file_name) as input_file:
        file_content = input_file
        for line in file_content:
            prev_tag = start_token
            line_number += 1
            if line_number % 3 == 1:
                continue
            if line_number % 3 == 0:
                word_line = line.split()
                continue
            word_index = 0
            for tag in line.split():
                word = word_line[word_index]
                word_set.add(word)
                add_to_bigram_dict(tag_bigram, prev_tag, tag)
                inc_dict_value(current_dict=word_tag_bigram, key=(tag, word))
                N += 1
                prev_tag = tag
                word_index += 1
                word_count[word] = 1 + word_count.get(word, 0)
            add_to_bigram_dict(tag_bigram, prev_tag, end_token)
    return {
        'N': N,
        'V_tag': len(tag_bigram) - 1,
        'tag_bigram': tag_bigram,
        'word_tag_bigram': word_tag_bigram, 
        'V_words': len(word_set),
        'word_count': word_count,
    }


def handle_unknown_word_tag_combinations(bigram_data, threshold=1, unk_token='<unk>'):
    under_threshold = set()
    smoothed_bigram = {}
    for word_tag_combo in bigram_data['word_tag_bigram']:
        if bigram_data['word_tag_bigram'][word_tag_combo] <= threshold:
            smoothed_bigram[(word_tag_combo[0], unk_token)] = bigram_data['word_tag_bigram'][word_tag_combo] + smoothed_bigram.get((word_tag_combo[0], unk_token), 0)
        else:
            smoothed_bigram[word_tag_combo] = bigram_data['word_tag_bigram'][word_tag_combo]
    bigram_data['word_tag_bigram'] = smoothed_bigram


def calculate_transition_probabilities(bigram_data, tags_possible, k):
    tag_prob_matrix = []
    V = len(tags_possible)
    for prev_tag in range(len(tags_possible)):
        tag_prob_matrix.append([])
        denominator = bigram_data['tag_bigram'][tags_possible[prev_tag]]['count'] + float(k) * V
        for next_tag in range(len(tags_possible)):
            numerator = float(k) + (bigram_data['tag_bigram'][tags_possible[prev_tag]]['successors'][tags_possible[next_tag]] if tags_possible[next_tag] in bigram_data['tag_bigram'][tags_possible[prev_tag]]['successors'] else 0)
            transition_probability = float(numerator) / denominator
            tag_prob_matrix[prev_tag].append(transition_probability)
    return tag_prob_matrix


def calculate_start_transition_probabilities(bigram_data, tags_possible, k):
    start_token = '<s>'
    V = len(tags_possible)
    start_transition_probabilities = []
    denom = float(bigram_data['tag_bigram'][start_token]['count']) + k * V
    default_transition_probability = float(k) / denom
    for i in range(len(tags_possible)):
        if tags_possible[i] in bigram_data['tag_bigram'][start_token]['successors']:
            start_transition_probabilities.append(float((bigram_data['tag_bigram'][start_token]['successors'][tags_possible[i]]) + k) / denom)
        else:
            start_transition_probabilities.append(default_transition_probability)
    return start_transition_probabilities


def calculate_word_probabilities(tags_possible, words, bigram_data, k, unk_token='<unk>'):
    words_in_line = words
    print('breakpoint_2')
    print(words_in_line)
    word_probabilities = []
    V = bigram_data['V_words']
    for i in range(len(tags_possible)):
        word_probabilities.append([])
        denom = bigram_data['tag_bigram'][tags_possible[i]]['count']
        for j in range(len(words_in_line)):
            if (tags_possible[i], words_in_line[j]) in bigram_data['word_tag_bigram']:
                #denom = bigram_data['word_count'][words_in_line[j]]
                word_probabilities[i].append(float(bigram_data['word_tag_bigram'][(tags_possible[i], words_in_line[j])]) / denom)
            else:
                word_probabilities[i].append(0.1e-15)
                #word_probabilities[i].append(float(bigram_data['word_tag_bigram'][(tags_possible[i], unk_token)])/ denom)
    return word_probabilities


def viterbi_2(line, word_index, tags_possible, word_probabilities, transition_probabilities, start_transition_probabilities):
    print ('breakpoint_1')
    print (tags_possible)
    print (transition_probabilities)
    print (word_probabilities)
    frontier = []
    back_pointer = []
    for i in range(len(tags_possible)):
        frontier.append([])
        back_pointer.append([])
        for j in range(len(line)):
            frontier[i].append(0.0)
            back_pointer[i].append(-1)
    for i in range(len(tags_possible)):
        frontier[i][0] = start_transition_probabilities[i] * word_probabilities[i][0]
    ## Initializations end!
    for word_index in range(1, len(line)):
        for tag_index in range(len(tags_possible)):
            best_score = 0.0
            best_prev_tag_index = 2
            for prev_tag_index in range(len(tags_possible)):
                current_score = float(frontier[prev_tag_index][word_index - 1]) * transition_probabilities[prev_tag_index][tag_index] * word_probabilities[tag_index][word_index]
                if current_score > best_score:
                    best_score = current_score
                    best_prev_tag_index = prev_tag_index
            frontier[tag_index][word_index] = best_score
            back_pointer[tag_index][word_index] = best_prev_tag_index
    ## Check for max last word tag score
    best_end_tag_index = 2
    best_end_score = 0.0
    last_word_index = len(line) - 1
    for tag_index in range(len(tags_possible)):
        if best_end_score < frontier[tag_index][last_word_index]:
            best_end_score = frontier[tag_index][last_word_index]
            best_end_tag_index = tag_index
    ## Back-track using best_end_index:
    result = [best_end_tag_index]
    word_index = last_word_index
    while word_index > 0:
        curr_tag_index = back_pointer[result[0]][word_index]
        result = [curr_tag_index] + result
        word_index -= 1
    print('breakpoint_3')
    print(frontier)
    print(back_pointer)
    return result


# Find paragraph wise precision and recall from a given validation_file and return summation of all, for a given k value
def predict(validation_file, bigram_data, k):
    predicted_count = 1
    correctly_predicted_count = 0
    predicted_where_expected_count = 0
    expected_predictions_count = 1
    start_token='<s>'
    end_token='</s>'
    unk_token='<unk>'
    line_number = 0
    tags_possible = []
    for tag in bigram_data['tag_bigram']:
        if tag != start_token and tag != unk_token:
            tags_possible.append(tag)
    transition_probabilities = calculate_transition_probabilities(bigram_data=bigram_data, tags_possible=tags_possible, k=k)
    start_transition_probabilities = calculate_start_transition_probabilities(bigram_data=bigram_data, tags_possible=tags_possible, k=k)
    #unseen_word_probabilities = calculate_unseen_word_probabilities(bigram_data, tags_possible, k)
    result_list = []
    with open(validation_file) as input_file:
        file_content = input_file
        for line in file_content:
            #print('processing line: {line}'.format(line=line_number))
            line = line.rstrip('\n')
            if line_number % 3 == 0:
                words = line.split('\t')
            elif line_number % 3 == 2:
                tags = line.split('\t')
                word_index = 0
                word_probabilities = calculate_word_probabilities(tags_possible, words, bigram_data, k)
                result = viterbi_2(
                    line=words,
                    word_index=0,
                    tags_possible=tags_possible,
                    word_probabilities=word_probabilities,
                    transition_probabilities=transition_probabilities,
                    start_transition_probabilities=start_transition_probabilities,
                )
                print ('result is {result}'.format(result=result))
                result_list.append(result)
            line_number += 1
    return (result_list, tags_possible)


def get_tag(tag):
    return tag if tag == 'O' else tag[2:]


def parse_results(result_list, tags_possible):
    result = {
        'ORG': '',
        'MISC': '',
        'PER': '',
        'LOC': '',
    }
    word_number = 0
    prev_tag = None
    prev_tag_start = None
    for i in range(len(result_list)):
        for j in range(len(result_list[i])):
            tag = get_tag(tags_possible[result_list[i][j]])
            if tag != prev_tag:
                if prev_tag and prev_tag != 'O':
                   result[prev_tag] = '{prev_tag_str} {prev_tag_start}-{prev_tag_end}'.format(
                        prev_tag_str=result[prev_tag],
                        prev_tag_start=str(prev_tag_start),
                        prev_tag_end=str(int(word_number) - 1),
                    )
                prev_tag = tag
                prev_tag_start = word_number
            word_number += 1
    return result


# Find paragraph wise precision and recall from a given validation_file and return summation of all, for a given k value
def find_bigram_precision_recall_fmeasure(validation_file, bigram_data, k):
    predicted_count = 0.001
    correctly_predicted_count = 0
    predicted_where_expected_count = 0
    expected_predictions_count = 0.001
    start_token='<s>'
    end_token='</s>'
    unk_token='<unk>'
    line_number = 0
    tags_possible = []
    for tag in bigram_data['tag_bigram']:
        if tag != start_token and tag != unk_token:
            tags_possible.append(tag)
    transition_probabilities = calculate_transition_probabilities(bigram_data=bigram_data, tags_possible=tags_possible, k=k)
    start_transition_probabilities = calculate_start_transition_probabilities(bigram_data=bigram_data, tags_possible=tags_possible, k=k)
    print('debugging')
    print(bigram_data)
    print(tags_possible)
    print(transition_probabilities)
    print(start_transition_probabilities)
    #print(bigram_data['word_tag_bigram'][('O', 'Korea')])
    print(bigram_data['word_tag_bigram'][('I-LOC', 'Korea')])
    print(bigram_data['word_count']['Korea'])
    with open(validation_file) as input_file:
        file_content = input_file
        for line in file_content:
            #print('processing line: {line}'.format(line=line_number))
            line = line.rstrip('\n')
            if line_number % 3 == 0:
                words = line.split('\t')
            elif line_number % 3 == 2:
                tags = line.split('\t')
                word_probabilities = calculate_word_probabilities(tags_possible, words, bigram_data, k)
                print('debug_2')
                print(word_probabilities)
                result = viterbi_2(
                    line=words,
                    word_index=0,
                    tags_possible=tags_possible,
                    word_probabilities=word_probabilities,
                    transition_probabilities=transition_probabilities,
                    start_transition_probabilities=start_transition_probabilities,
                    #frontier=[],
                    #back_pointer=[],
                )
                print ('result is {result}'.format(result=result))
                predicted_tags = result
                word_index = 0
                while word_index < len(words):
                    word = words[word_index]
                    tag = tags[word_index]
                    predicted_tag = tags_possible[predicted_tags[word_index]]
                    #print(predicted_tag)
                    if tag != 'O':
                        expected_predictions_count += 1
                    if tag == predicted_tag and tag != 'O':
                        correctly_predicted_count += 1
                    if predicted_tag != 'O':
                        predicted_count += 1
                    if predicted_tag != 'O' and tag != 'O':
                        predicted_where_expected_count += 1
                    word_index += 1
            line_number += 1
    precision = float(correctly_predicted_count)/predicted_count
    recall = float(predicted_where_expected_count)/expected_predictions_count
    fmeasure = 2*precision*recall/(precision+recall)
    print (tags_possible)
    return {
        'precision': precision,
        'recall': recall,
        'fmeasure': fmeasure,
    }


############### Test code here ###################

input_file_name='Project2_fall2018/Project2_fall2018/train_split.txt'
validation_file_name='Project2_fall2018/Project2_fall2018/validation_split.txt'

bigram_data = process_corpus(input_file_name)
handle_unknown_word_tag_combinations(bigram_data, threshold=1)
#print(bigram_data)
result_list=predict(validation_file=validation_file_name, bigram_data=bigram_data, k=1)
res = parse_results(result_list=result_list[0], tags_possible=result_list[1])
#print(res)
#result=find_bigram_precision_recall_fmeasure(validation_file=validation_file_name, bigram_data=bigram_data, k=0.01)
#print result




