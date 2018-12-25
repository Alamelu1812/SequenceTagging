from sklearn.linear_model import LogisticRegression
import pickle
import gensim
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from sklearn.linear_model import SGDClassifier




def get_vectors_for_matching_words(words, vector_file_name):
    vector_dict={}
    unk_token='<unk>'
    unk_vector=None
    with open(vector_file_name) as vector_file:
        file_content = vector_file
        for line in file_content:
            vector = line.split()
            if vector[0] in words:
                vector_dict[vector[0]] = [float(value) for value in vector[1:]]
            elif not unk_vector:
                unk_vector = [float(value) for value in vector[1:]]
    vector_dict[unk_token]=unk_vector
    return vector_dict


def get_word_to_vec_embeddings_after_saving(train_file_name, test_file_name, vector_file_name, pickle_file_name):
    words = get_all_words_from_file(train_file_name)
    words.update(get_all_words_from_file(test_file_name))
    model = gensim.models.KeyedVectors.load_word2vec_format(vector_file_name, binary=True)
    magic_unused_word = "kuchipudi"
    word_vectors = model.wv
    unk_vector = word_vectors[magic_unused_word]
    del model
    required_word_vectors = {}
    for word in words:
        if word in word_vectors:
            required_word_vectors[word] = word_vectors[word]
        else:
            required_word_vectors[word] = unk_vector
    pickle.dump(required_word_vectors, open(pickle_file_name, "wb"))
    return required_word_vectors


def get_word_embeddings_after_saving(input_file_name, vector_file_name, pickle_file_name):
    words = get_all_words_from_file(input_file_name)
    vector_dict=get_vectors_for_matching_words(words, vector_file_name)
    pickle.dump(vector_dict, open(pickle_file_name, "wb"))
    return vector_dict


def is_numeric(word):
    try:
        float(word)
        return True
    except ValueError:
        return False


def get_feature_vectors_for_sentence(word_vectors, pos_tags_dict, sentence, sentence_pos_tags):
    zero_vector = [0 for current in range(0, 300)]
    window = [zero_vector, zero_vector, zero_vector, zero_vector]
    pos_window = [0, 0, 0, 0]
    tag_window = [0, 0, 0, 0]
    feature_list = []
    word_number = 0
    sentence_pos_tags = sentence_pos_tags.split()
    after_apostrophe_to_use = 0
    # Initialize transfer-flags
    is_after_comma = 0
    is_after_colon = 0
    is_after_hyphen = 0
    is_after_period = 0
    is_after_brace_open = 0
    is_after_brace_close = 0
    is_after_apostrophe = 0
    for word in sentence.split():
        # Punctuation checks for current word
        is_comma = 1 if word == "," else 0
        is_colon = 1 if word == ":" else 0
        is_hyphen = 1 if word == "-" else 0
        is_period = 1 if word == "." or word == '!' or word== "?" or word == ";" else 0
        is_brace_open = 1 if word == "(" or word == "[" or word == "{" else 0
        is_brace_close = 1 if word == ")" or word == "]" or word == "}" else 0
        is_apostrophe = 1 if word == "'" else 0
        # Set positional flags
        is_int = 1 if word.isdigit() else 0
        is_decimal = 1 if is_numeric(word) else 0
        has_digit = 1 if any(char.isdigit() for char in word) else 0
        has_apostrophe = 1 if any(char == "'" for char in word) else 0
        has_hyphen = 1 if any(char == "-" for char in word) else 0
        is_capitalized = 1 if len(word) > 0 and word[0].isupper() else 0
        flag_features = [is_after_comma, is_after_colon, is_after_hyphen, is_after_period, is_after_brace_open, is_after_brace_close, is_after_apostrophe, is_comma, is_colon, is_hyphen, is_period, is_brace_open, is_brace_close, is_apostrophe, is_int, is_decimal, has_digit, has_apostrophe, has_hyphen, is_capitalized]
        # After all flags have been evaulated, use word_embeddings and flags to create feature 
        word = word.lower()
        pos_tag = pos_tags_dict[sentence_pos_tags[word_number].lower()]
        window = window[1:] + [word_vectors[word]]
        pos_window = pos_window[1:] + [pos_tag]
        feature = []
        # Append word_embeddings to feature
        for word_vector in window:
            for val in word_vector:
                feature.append(val)
        for flag in flag_features:
            feature.append(flag)
        for pos_tag in pos_window:
            feature = feature + [pos_tag]
        # Append flags to feature
        #print ("appending feature with len {mylen}".format(mylen=len(feature)))
        feature_list.append(feature)
        after_apostrophe_to_use = 0 if is_after_apostrophe == 1 else after_apostrophe_to_use
        word_number += 1
        # Transfer flags
        is_after_comma = is_comma
        is_after_colon = is_colon
        is_after_hyphen = is_hyphen
        is_after_period = is_period
        is_after_brace_open = is_brace_open
        is_after_brace_close = is_brace_close
        is_after_apostrophe = is_apostrophe
    return feature_list


def get_all_words_from_file(input_file_name, line_mod=1):
    words = set()
    with open(input_file_name) as file_content:
        line_number = 0
        for line in file_content:
            line_number += 1
            if line_number % 3 != line_mod:
                continue
            words_in_line = line.split()
            words.update([word.lower() for word in words_in_line])
    return words


def build_pos_tags_dict(train_file_name, test_file_name):
    pos_tags_dict = {}
    pos_tags = get_all_words_from_file(input_file_name=train_file_name, line_mod=2)
    pos_tags.update(get_all_words_from_file(input_file_name=test_file_name, line_mod=2))
    pos_key = 0
    for pos in pos_tags:
        if pos not in pos_tags_dict:
            pos_tags_dict[pos] = pos_key
            pos_key += 1
    return pos_tags_dict


def get_and_save_pos_tags(train_file_name, test_file_name, pos_file_name):
    pos_tags_dict = build_pos_tags_dict(train_file_name, test_file_name)
    pickle.dump(pos_tags_dict, open(pos_file_name, "wb"))
    return pos_tags_dict


def get_pickled_data(pickle_file_name):
    return pickle.load(open(pickle_file_name, "rb"))


def build_and_save_iob_tags_dict(train_file_name, iob_file_name):
    iob_tags_dict = {}
    iob_tags = get_all_words_from_file(input_file_name=train_file_name, line_mod=0)
    iob_key = 0
    for iob in iob_tags:
        if iob not in iob_tags_dict:
            iob_tags_dict[iob] = iob_key
            iob_key += 1
    pickle.dump(iob_tags_dict, open(iob_file_name, "wb"))
    return iob_tags_dict


#train_file_name="Project2_fall2018/Project2_fall2018/train_sample.txt"
train_file_name="Project2_fall2018/Project2_fall2018/train.txt"
validation_file_name="Project2_fall2018/Project2_fall2018/validation_split.txt"
test_file_name="Project2_fall2018/Project2_fall2018/test.txt"
vector_file_name="./GoogleNews-vectors-negative300.bin"
word_embedding_file_name="encountered_word_vectors.txt"
pos_file_name="pos.txt"
iob_file_name="iob.txt"
train_data_file="train_data.txt" # for hinge loss - trained file
#train_data_file="log_train_data.txt" # for hinge loss - trained file

### Create and save word embeddings and pos dicts to files
word_vectors=get_word_embeddings_after_saving(input_file_name="glove.840B.300d.txt", vector_file_name, pickle_file_name)
print('Calculating new word vectors')
word_vectors=get_word_to_vec_embeddings_after_saving(train_file_name, test_file_name, vector_file_name, word_embedding_file_name)
print('Calculated new word vectors')
pos_tags_dict=get_and_save_pos_tags(train_file_name, test_file_name, pos_file_name)
iob_tags_dict=build_and_save_iob_tags_dict(train_file_name, iob_file_name)


### Retrieve saved word embeddings and pos dicts from files
word_vectors=get_pickled_data(word_embedding_file_name)
pos_tags_dict=get_pickled_data(pos_file_name)
iob_tags_dict=get_pickled_data(iob_file_name)

#print(word_vectors)
#print(word_vectors.keys())


#print(get_all_words_from_file(input_file_name))


feature_list = get_feature_vectors_for_sentence(word_vectors, pos_tags_dict, sentence, sentence_pos_tags)


def get_tags_list(tags_dict, tag_sentence):
    tags_list = []
    for tag in tag_sentence.split():
        tags_list.append(tags_dict[tag.lower()])
    return tags_list


def get_features_and_tags_for_all_sentences(word_vectors, pos_tags_dict, iob_tags_dict, train_file_name, train_data_file):
    feature_list = []
    tag_list = []
    sgd_model = SGDClassifier(loss="log")
    all_classes = iob_tags_dict.values()
    with open(train_file_name) as file_content:
        sentence = ""
        pos_line = ""
        sentence_pos_tags = ""
        line_number = 0
        for line in file_content:
            line_number += 1
            if line_number % 3 == 1:
                sentence = line
            elif line_number % 3 == 2:
                sentence_pos_tags = line
            else:
                tag_sentence = line
                sentence_feature_list = get_feature_vectors_for_sentence(word_vectors, pos_tags_dict, sentence, sentence_pos_tags)
                for sentence_feature in sentence_feature_list:
                    feature_list.append(sentence_feature)
                sentence_tag_list = get_tags_list(iob_tags_dict, tag_sentence)
                for sentence_tag in sentence_tag_list:
                    tag_list.append(sentence_tag)
            if line_number % 1000 == 0 or line_number == 41999:
                print('Training SGD upto line {line_number}'.format(line_number=line_number))
                #print(len(feature_list))
                #print(len(feature_list[0]))
                #print(len(tag_list))
                sgd_model.partial_fit(feature_list, tag_list, classes=all_classes)
                feature_list = []
                tag_list = []
                print('Trained SGD upto line {line_number}'.format(line_number=line_number))
                if line_number == 41999:
                    break
        print('Finished Training SGD upto line {line_number}'.format(line_number=line_number))
        #sgd_model.partial_fit(feature_list, tag_list, classes=all_classes)
        #print('Trained SGD fully, upto line {line_number}'.format(line_number=line_number))
    pickle.dump(sgd_model, open(train_data_file, "wb"))



def get_predictions(word_vectors, pos_tags_dict, iob_tags_dict, test_file_name, sgd_model):
    iob_num_to_tags_dict = {}
    result = {}
    for key in iob_tags_dict.keys():
        iob_num_to_tags_dict[iob_tags_dict[key]] = key
    with open(test_file_name) as file_content:
        line_number = 0
        sentence = ""
        sentence_pos_tags = ""
        word_numbers = ""
        predictions = {}
        for line in file_content:
            line_number += 1
            if line_number % 3 == 1:
                sentence = line
                continue
            if line_number % 3 == 2:
                sentence_pos_tags = line
                continue
            word_numbers = line.split()
            feature_vectors_for_line = get_feature_vectors_for_sentence(word_vectors, pos_tags_dict, sentence, sentence_pos_tags)
            sentence_prediction = sgd_model.predict(feature_vectors_for_line)
            print('predicted result is {predictions}'.format(predictions=sentence_prediction))
            print('word_numbers {word_numbers}'.format(word_numbers=word_numbers))
            for word_index_in_sentence in range(0, len(sentence_prediction)):
                print('{word_number} - {tag}'.format(word_number=word_numbers[word_index_in_sentence], tag=sentence_prediction[word_index_in_sentence]))
                result[int(word_numbers[word_index_in_sentence])] = iob_num_to_tags_dict[sentence_prediction[word_index_in_sentence]]
    return result


#get_features_and_tags_for_all_sentences(word_vectors, pos_tags_dict, iob_tags_dict, train_file_name, train_data_file)
sgd_model=get_pickled_data(train_data_file)
result_dict = get_predictions(word_vectors, pos_tags_dict, iob_tags_dict, test_file_name, sgd_model)
print(result_dict)


def get_tag(tag):
    return tag if tag == 'O' else tag[2:].upper()


def parse_results(result_dict):
    result = {
        'ORG': '',
        'MISC': '',
        'PER': '',
        'LOC': '',
    }
    prev_tag = None
    prev_tag_start = None
    for word_number in result_dict.keys():
        tag = get_tag(result_dict[word_number])
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


parsed_results = parse_results(result_dict)
print(parsed_results)





