# NOTES FROM GPT
- simplifier - spaCy's textacy
Pretrained models 
- they are deep neural networks with multiple layers and attention mechanisms that learn to understand and generate text.
- BERT
- BART = better suited for summarising tasks
- We can finetune the models pretrained on legal corpora
  - Cornell Legal Information Institute (LII): LII provides a wide range of legal documents, including U.S. Supreme Court decisions, the United States Code, and more.
  - Legal Data Sets from Government Sources: Many governments provide access to legal documents, such as statutes, regulations, and court decisions, that can be used for training.
- Legal Research Databases: Legal research databases like Westlaw, LexisNexis, and PACER might provide access to a vast collection of legal documents.
- Open Access Legal Journals: Some legal journals and publications provide open access to legal articles and documents.
- Web Scraping: You can use web scraping to collect legal documents from publicly available legal websites and databases.

0-----
# bart + legal corpora maybe
Token Embeddings:

Depending on the size of your domain-specific corpus, you have two main options:
**Fine-tuning All Layers:** If you have a large corpus, you can fine-tune all layers of the pre-trained model. This allows the model to adapt to the new domain more thoroughly.
**Feature Extractor:** If your domain-specific corpus is relatively small, you can use the pre-trained model as a feature extractor. You feed your domain-specific data to the pre-trained model, and the embeddings from the model are used as features for a downstream task.

---------

Date 10/28 
- BART Facebook/hugging face - seq to seq model **Bidirectional and Auto-Regressive Transformers**
    - uses BERT (bi -enc + decode) and GPT (left to right decoder)
- **Challenges**
- Longer and exceeds token limit on most systems
    - Chunking, longformer, extractive + abstractive
- Domain independent gives better results than domain specific so far
- Score of ROGUE vs experts -> extractive preferred by practitioners
- **IDEAS**
- Legal dictionary for identifying keywords
- 
  - **Paper**
  - Unsupervised extractive => lex rank, dsdr, pacsum,
  - supervised extractive => summarunner, BERT-summ
  - supervised abstractive => BART, longformer
  - Domain specific methods => MMr, CaseSummariser, Gist
  - Extractive : LetSum, kmm, casesummariser, mmr
  - Abstractive: long (divide n conquer approach) and 
