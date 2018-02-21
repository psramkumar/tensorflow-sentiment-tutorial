import numpy as np


def _words_to_embedding_index(words, word2idx):
    """Converts each word in a list of words to its
    corresponding row in the embedding matrix.

    Parameters
    ----------
        words : list[unicode or str]

        word2idx : dict[unicode or str, int]

    Returns
    -------
        : list[int]
    """
    sentence_ids = []
    for word in words:
        if word in word2idx:
            sentence_ids.append(word2idx[word])
        else:
            sentence_ids.append(word2idx['<unk>'])
    return sentence_ids


def _add_filter_padding(sentence_ids, word2idx, filter_len=5):
    """Adds filter padding to the front and back of a sentence (list of
        word embedding indices)

    Parameters
    ----------
        sentence_ids : list[int]
            A list of word embedding indices corresponding with each index
            corresponding to a word in a sentence.

        word2idx : dict[unicode or str, int]

        filter_len : int
            Length of filter.

    Returns
    -------
        : list[int]
    """
    if filter_len > 1:
        filter_padding = (filter_len - 1) * [word2idx['<filter_padding>']]
    else:
        filter_padding = []
    return filter_padding + sentence_ids + filter_padding


def words_to_embedding_index_with_padding(words, word2idx, filter_len=5):
    """Converts each word to its corresponding
        row in the embedding matrix. Adds a filter padding to the front
        and back of the sentence.

    Parameters
    ----------
        words : list[unicode]
            The words to convert.

        word2idx : dict[unicode, int]
            Mapping of word to row of embedding matrix.

        filter_len : int
            Length of filter used.

    Returns
    -------
        : list[int]
            List of integer corresponding to the row of each word in the
            embedding matrix.
    """
    sentence_ids = _words_to_embedding_index(words,
                                             word2idx)
    padded_sentence_ids = _add_filter_padding(sentence_ids,
                                              word2idx,
                                              filter_len)

    return padded_sentence_ids


def pad_sequences(sequences, val, maxlen=68, padding='post',
                  truncating='pre', dtype=np.int32):
    """Transform a list of nb_samples sequences (lists of scalars) into
        a 2D Numpy array of shape  (nb_samples, nb_timesteps).
        nb_timesteps is either the maxlen argument if provided, or the
        length of the longest sequence otherwise.

    Parameters
    ----------
        sequences : list[list[int]]

        val : int or float
            Value to pad the sequences to the desired value

        maxlen : None or int
            Maximum sequence length, longer sequences are truncated and
            shorter sequences are padded with zeros at the beginning or end.

        padding : str
            One of 'pre' or 'post', pad either before or after each sequence.

        truncating : str
            One of 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence

        dtype : np.int32
            Data type of the Numpy array returned.

    Returns
    -------
        : np.ndarray[np.int32]
            2D Numpy array of shape (nb_samples, nb_timesteps)
    """

    if padding not in ['post', 'pre']:
        raise ValueError("Argument 'padding' must be of value 'pre'"
                         "or 'post'. However, '{}' was given.".format(
                             padding))

    if truncating not in ['post', 'pre']:
        raise ValueError("Argument 'truncating' must be of value 'pre'"
                         "or 'post'. However, '{}' was given.".format(
                             padding))

    if val is None:
        raise ValueError("Argument 'value' must not be None.")

    if maxlen is None:
        maxlen = np.max([len(sequence) for sequence in sequences])

    processed_sequences = []
    for sequence in sequences:
        seq_length = len(sequence)
        if seq_length > maxlen:
            if truncating == 'pre':
                processed_sequences.append(sequence[-maxlen:])
            else:
                processed_sequences.append(sequence[:maxlen])
        elif seq_length < maxlen:
            pad_sequence = (maxlen - seq_length) * [val]
            if padding == 'pre':
                processed_sequences.append(pad_sequence + sequence)
            else:
                processed_sequences.append(sequence + pad_sequence)
        else:
            processed_sequences.append(sequence)

    return np.asarray(processed_sequences, dtype=dtype)