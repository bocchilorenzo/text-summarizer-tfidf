# text-summarizer-tfidf
Simple text summarizer using TF-IDF.

## How to use
NOTE: You can skip steps 2, 3 and 4 if you already have UDpipe 1 and the models installed or if you want to use the NLTK sentence tokenizer instead of UDpipe's.

1. Clone the repository, and install the requirements:
```bash
pip install -r requirements.txt
```

2. Install UDpipe 1. You can find installation instructions on https://ufal.mff.cuni.cz/udpipe/1/install. In short, download the release from Github and install the binary (on Windows, copy the folder for either the 32bit or 64bit binary wherever you want and add its path to the PATH environment variable).

3. Download the zip with all the UDpipe models from http://hdl.handle.net/11234/1-3131

4. Create a folder named 'models' in the root directory of this repository, and extract the models from the zip in it.

5. Import the summarizer:
```python
from summarizer import TFIDFSummarizer
```

6. Finally, create a Summarizer object and call the summarize method:
```python
summ = TFIDFSummarizer(language="italian", norm="l2", tokenizer="udpipe")
result = summ.summarize([text1, text2, ...])
```

The summarizer vectorizes the text of all the documents. The result contains a list of tuples, each containing a list of the sentences from the original text and a list of scores for each sentence.