# text-summarizer-tfidf
Simple text summarizer using TF-IDF.

## How to use

First, install the dependencies:
```bash
pip install -r requirements.txt
```

Then, import the summarizer:
```python
from summarizer import TFIDFSummarizer
```

Finally, create a Summarizer object and call the summarize method:
```python
summarizer = TFIDFSummarizer(lang="italian", norm="l2")
result = summarizer.summarize([text1, text2, ...])
```

The summarizer vectorizes the text of all the documents. The result contains a list of tuples, each containing a list of the sentences from the original text and a list of scores for each sentence.