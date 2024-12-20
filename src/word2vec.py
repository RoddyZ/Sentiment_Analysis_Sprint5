from typing import List

import numpy as np
from gensim.models.word2vec import Word2Vec


def vectorizer(
    corpus: List[List[str]], model: Word2Vec, num_features: int = 100
) -> np.ndarray:
    """
    This function takes a list of tokenized text documents (corpus) and a pre-trained
    Word2Vec model as input, and returns a matrix where each row represents the
    vectorized form of a document.

    Args:
        corpus : list
            A list of text documents that needs to be vectorized.

        model : Word2Vec
            A pre-trained Word2Vec model that will be used to vectorize the corpus.

        num_features : int
            The size of the vector representation of each word. Default is 100.

    Returns:
        corpus_vectors : numpy.ndarray
            A 2D numpy array where each row represents the vectorized form of a
            document in the corpus.
    """
    # TODO
    # Initialize an empty list to store document vectors
    corpus_vectors = []

    for document in corpus:
        # Initialize an empty array to store word vectors for this document
        word_vectors = []

        for word in document:
            # Check if the word exists in the Word2Vec model's vocabulary
            if word in model.wv.key_to_index:
                word_vectors.append(model.wv[word])  # Get the word vector

        if word_vectors:
            # Compute the average vector for this document
            document_vector = np.mean(word_vectors, axis=0)
        else:
            # If no word vectors are found for the document, create a zero vector
            document_vector = np.zeros(num_features)

        corpus_vectors.append(document_vector)

    # Convert the list of document vectors to a 2D numpy array
    return np.array(corpus_vectors)
