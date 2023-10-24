# sankshepika-mlpro
Legal document summariser

## STEPS
- Data
**Annotated data available? - legal docs and their actual summaries are not available right?**
- preprocessing
  - **Study about stopwords, stemming, sentence segmentation**
- Feature extraction
  - **TF-IDF, embeddings (own or trained models)**
- Models
  - **Extractive** - choose most important lines as such.
    - **_Textrank, pagerank, BERT based_**
  - **Abstractive** - generate summary in own words like GPT
    - **_Seq2Seq models like LSTM, transformer based, GPT (access to chatgpt)_**
- Training
- Evaluation
  - **How to test our summary quality** - 
  _ROUGE - **Recall oriented understudy for gisting eval**_
  _human evaluation_
