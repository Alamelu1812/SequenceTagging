max_tag = 'MAX_TAG'
max_count_tag = 'MAX_TAG_COUNT'

def parse_for_word_tags(file_name):
    line_number = 0
    word_tag_counts = {}
    with open(file_name) as lines:
        for line in lines:
            line = line.rstrip('\n')
            if line_number % 3 == 0:
                words = line.split('\t')
            elif line_number % 3 == 2:
                tags = line.split('\t')
                word_index = 0
                while word_index < len(words):
                    word = words[word_index]
                    tag = tags[word_index]
                    if word not in word_tag_counts:
                        word_tag_counts[word] = {max_count_tag: 0}
                    word_tag_counts[word][tag] = word_tag_counts[word].get(tag, 0) + 1
                    if word_tag_counts[word][max_count_tag] < word_tag_counts[word][tag]:
                        word_tag_counts[word][max_count_tag] = word_tag_counts[word][tag]
                        word_tag_counts[word][max_tag] = tag
                    word_index += 1
            line_number += 1
    return word_tag_counts


def get_tag(tag):
    return tag if tag == 'O' else tag[2:]


def predict_test_tags(test_file_name, word_tag_counts):
    result = {
        'ORG': '',
        'MISC': '',
        'PER': '',
        'LOC': '',
    }
    line_number = 0
    with open(test_file_name) as lines:
        for line in lines:
            line = line.rstrip('\n')
            if line_number % 3 == 0:
                words = line.split('\t')
            elif line_number % 3 == 2:
                word_numbers = line.split(' ')
                prev_tag = None
                prev_tag_start = None
                word_index = 0
                while word_index < len(words):
                    word = words[word_index]
                    word_number = word_numbers[word_index]
                    word_index += 1
                    tag = 'O' if word not in word_tag_counts else get_tag(word_tag_counts[word][max_tag])
                    if tag != prev_tag:
                        if prev_tag and prev_tag != 'O':
                            result[prev_tag] = '{prev_tag_str} {prev_tag_start}-{prev_tag_end}'.format(
                                prev_tag_str=result[prev_tag],
                                prev_tag_start=str(prev_tag_start),
                                prev_tag_end=str(int(word_number) - 1),
                            )
                        prev_tag = tag
                        prev_tag_start = word_number
            line_number += 1
    return result


## Run steps:

file_name = 'Project2_fall2018/Project2_fall2018/train.txt'
test_file_name = 'Project2_fall2018/Project2_fall2018/test.txt'

word_tag_counts = parse_for_word_tags(file_name)
prediction_results = predict_test_tags(test_file_name, word_tag_counts)





