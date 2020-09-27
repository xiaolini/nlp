import numpy as np
import pickle
import os
import copy

CODES = {'<PAD>': 0, '<EOS>': 1, '<UNK>': 2, '<GO>': 3 }


def load_data(path):
    """
    Load Dataset from File
    """
    input_file = os.path.join(path)
    with open(input_file, 'r') as f:
        return f.read()

def text_to_ids(source_text,target_text,source_vocab_to_int,target_vocab_to_int):
    """
        Convert source and target text to proper word ids
        :param source_text: String that contains all the source text.
        :param target_text: String that contains all the target text.
        :param source_vocab_to_int: Dictionary to go from the source words to an id
        :param target_vocab_to_int: Dictionary to go from the target words to an id
        :return: A tuple of lists (source_id_text, target_id_text)
        """

    source_id_text = [[source_vocab_to_int.get(word, source_vocab_to_int['<UNK>'])
                       for word in line.split()]
                      for line in source_text.split('\n')]
    target_id_text = [[target_vocab_to_int.get(word, source_vocab_to_int['<UNK>'])
                       for word in line.split()] + [target_vocab_to_int['<EOS>']]
                      for line in target_text.split('\n')]

    return (source_id_text,target_id_text)

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    """
    vocab = set(text.split())
    vocab_to_int = copy.copy(CODES)  #字典的浅复制

    for v_i, v in enumerate(vocab, len(CODES)):
        vocab_to_int[v] = v_i

    int_to_vocab = {v_i: v for v, v_i in vocab_to_int.items()}

    return vocab_to_int, int_to_vocab

def preprocess_and_save_data(source_path, target_path, text_to_ids):
    """
    Preprocess Text Data.  Save to to file.
    """
    # Preprocess
    source_text = load_data(source_path)
    target_text = load_data(target_path)

    view_sentence_range = (0, 10)

    print('Dataset Stats')
    print(
        'Roughly the number of unique words cn: {}'.format(len({word: None for word in source_text.split()})))  # 13286
    print(
        'Roughly the number of unique words en: {}'.format(len({word: None for word in target_text.split()})))  # 11869
    sentences = source_text.split('\n')
    word_counts = [len(sentence.split()) for sentence in sentences]
    print('Number of sentences: {}'.format(len(sentences)))  # 6834
    print('Average number of words in a sentence: {}'.format(np.average(word_counts)))  # 22.4378

    print()
    print('Chinese sentences {} to {}:'.format(*view_sentence_range))
    print('\n'.join(source_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
    print()
    print('English sentences {} to {}:'.format(*view_sentence_range))
    print('\n'.join(target_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))

    source_text = source_text.lower()
    target_text = target_text.lower()
    source_vocab_to_int, source_int_to_vocab = create_lookup_tables(source_text)
    target_vocab_to_int, target_int_to_vocab = create_lookup_tables(target_text)

    source_text, target_text = text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int)

    # Save Data
    with open('preprocess.p', 'wb') as out_file:
        pickle.dump((
            (source_text, target_text),
            (source_vocab_to_int, target_vocab_to_int),
            (source_int_to_vocab, target_int_to_vocab)), out_file)


source_path = './data/cn.txt'
target_path = './data/en.txt'

#保存预处理后的数据用于训练
preprocess_and_save_data(source_path,target_path,text_to_ids)

