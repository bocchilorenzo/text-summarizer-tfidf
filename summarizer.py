from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer, sent_tokenize
import numpy as np
from re import sub
import warnings

warnings.simplefilter("ignore", category=RuntimeWarning)


class TFIDFSummarizer:
    """
    A class to summarize text using TF-IDF.
    
    :param lang: The language of the text to summarize.
    :param norm: The norm to use for the TF-IDF vectorizer.
    """

    def __init__(self, lang="italian", norm="l2"):
        self.lang = lang
        self.norm = norm
        self.stop = stopwords.words(self.lang)

    def _tokenize(self, text):
        sentence = TreebankWordTokenizer().tokenize(text.lower())
        temp = [i for i in sentence if i not in self.stop]
        return " ".join(temp)

    def _tfidf(self, docs):
        vectorizer = TfidfVectorizer(use_idf=True, norm=self.norm)
        vectorizer_vectors = vectorizer.fit_transform(docs)
        return vectorizer_vectors, vectorizer

    def summarize(self, docs):
        """
        Summarize a list of documents.

        :param docs: A list of documents to summarize.
        :return: A list of tuples, each containing a list of the sentences from the original text and a list of scores for each sentence.
        """
        tokenized_docs = [self._tokenize(doc) for doc in docs]
        tfidf_vectorizer_vectors, tfidf_vectorizer = self._tfidf(tokenized_docs)
        feature_indices = tfidf_vectorizer.vocabulary_
        text_counter = 0
        to_return = []
        for doc in docs:
            og_text = [
                sub(" +", " ", s.replace("\n", " ")).strip()
                for s in sent_tokenize(doc, language=self.lang)
            ]
            sentences = sent_tokenize(doc, "italian")
            fixed_sentences = [
                sub(" +", " ", s.replace("\n", " ")).strip() for s in sentences
            ]
            sent_eval = []
            row_vec = tfidf_vectorizer_vectors.getrow(text_counter).toarray()
            for s in fixed_sentences:
                sentence = TreebankWordTokenizer().tokenize(s.lower())
                temp = [i for i in sentence if i not in self.stop]
                sent_score = []
                for word in temp:
                    sent_score.append(
                        row_vec[0][feature_indices[word]]
                        if word in feature_indices
                        else 0
                    )
                sent_score = np.array(sent_score, dtype=np.float64)
                sent_score[sent_score == 0] = np.nan
                sent_eval.append(np.nanmean(sent_score))
            to_return.append((og_text, sent_eval))
            text_counter += 1
        return to_return
