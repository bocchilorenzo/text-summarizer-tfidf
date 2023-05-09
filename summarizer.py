from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from nltk.data import load
from nltk import download
download("punkt")
download("stopwords")
import ufal.udpipe
import numpy as np
from re import sub
from os import path
from yaml import safe_load
from copy import deepcopy
import warnings

warnings.simplefilter("ignore", category=RuntimeWarning)

# https://github.com/ufal/udpipe/tree/master/bindings/python/examples
class Model:
    def __init__(self, path):
        """Load given model."""
        self.model = ufal.udpipe.Model.load(path)
        if not self.model:
            raise Exception("Cannot load UDPipe model from file '%s'" % path)

    def tokenize(self, text):
        """Tokenize the text and return list of ufal.udpipe.Sentence-s."""
        tokenizer = self.model.newTokenizer(self.model.DEFAULT)
        if not tokenizer:
            raise Exception("The model does not have a tokenizer")
        return self._read(text, tokenizer)

    def read(self, text, format):
        """Load text in the given format (conllu|horizontal|vertical) and return list of ufal.udpipe.Sentence-s."""
        input_format = ufal.udpipe.InputFormat.newInputFormat(format)
        if not input_format:
            raise Exception("Cannot create input format '%s'" % format)
        return self._read(text, input_format)

    def _read(self, text, input_format):
        input_format.setText(text)
        error = ufal.udpipe.ProcessingError()
        sentences = []

        sentence = ufal.udpipe.Sentence()
        while input_format.nextSentence(sentence, error):
            sentences.append(sentence)
            sentence = ufal.udpipe.Sentence()
        if error.occurred():
            raise Exception(error.message)

        return sentences

    def tag(self, sentence):
        """Tag the given ufal.udpipe.Sentence (inplace)."""
        self.model.tag(sentence, self.model.DEFAULT)

    def parse(self, sentence):
        """Parse the given ufal.udpipe.Sentence (inplace)."""
        self.model.parse(sentence, self.model.DEFAULT)

    def write(self, sentences, format):
        """Write given ufal.udpipe.Sentence-s in the required format (conllu|horizontal|vertical)."""

        output_format = ufal.udpipe.OutputFormat.newOutputFormat(format)
        output = ""
        for sentence in sentences:
            output += output_format.writeSentence(sentence)
        output += output_format.finishDocument()

        return output

    def write_list(self, sentences):
        """Write given ufal.udpipe.Sentence-s in an iterable list."""

        output_format = ufal.udpipe.OutputFormat.newOutputFormat("horizontal")
        output = [output_format.writeSentence(sentence).strip() for sentence in sentences]
        
        return output


class TFIDFSummarizer:
    """
    A class to summarize text using TF-IDF.
    
    :param lang: The language of the text to summarize.
    :param norm: The norm to use for the TF-IDF vectorizer.
    :param tokenizer: The tokenizer to use. Either "udpipe" or "nltk".
    """

    def __init__(self, language="italian", norm="l2", tokenizer="udpipe"):
        self.lang = language
        self.norm = norm
        self.stop = stopwords.words(self.lang)
        with open("models.yml", "r") as f:
            self.model_configs = safe_load(f)
        if tokenizer == "udpipe":
            self.tokenizer_mode = "udpipe"
            self.sent_tokenizer = Model(
                path.join("./models", self.model_configs[language] + ".udpipe")
            )
        elif tokenizer == "nltk":
            self.tokenizer_mode = "nltk"
            self.sent_tokenizer = load(f"tokenizers/punkt/{language}.pickle")
        else:
            raise ValueError("Invalid tokenizer")

    def _tokenize(self, text):
        sentence = TreebankWordTokenizer().tokenize(text.lower())
        return [i for i in sentence if i not in self.stop]

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
        tokenized_docs = [" ".join(self._tokenize(doc)) for doc in docs]
        tfidf_vectorizer_vectors, tfidf_vectorizer = self._tfidf(tokenized_docs)
        feature_indices = tfidf_vectorizer.vocabulary_
        text_counter = 0
        to_return = []

        for doc in docs:
            # Save the original text
            og_text = [
                sub(" +", " ", s.replace("\n", " ")).strip()
                for s in self.sent_tokenizer.write_list(self.sent_tokenizer.tokenize(doc))
            ]
            # Make a copy of the original text to be modified
            fixed_sentences = deepcopy(og_text)
            sent_eval = []
            
            # Extract the TF-IDF score for each word in each sentence
            row_vec = tfidf_vectorizer_vectors.getrow(text_counter).toarray()
            for s in fixed_sentences:
                temp = self._tokenize(s)
                sent_score = []
                for word in temp:
                    sent_score.append(
                        row_vec[0][feature_indices[word]]
                        if word in feature_indices
                        else 0
                    )
                # Where the score is 0, replace it with NaN so it does not influence the mean of the sentence
                sent_score = np.array(sent_score, dtype=np.float64)
                sent_score[sent_score == 0] = np.nan
                sent_eval.append(np.nanmean(sent_score))
            to_return.append((og_text, sent_eval))
            text_counter += 1
        return to_return
