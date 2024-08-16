# Sankshepika

## Problem statement
Legal texts often contain complex language, specialized terms, and long sections, which can overwhelm people without legal knowledge. 
These documents are condensed into clear, straightforward summaries by summarizers. Key points are extracted, allowing non-experts to understand vital details without going into the complexities. 
Moreover, time is saved, and the need for costly legal consultations is reduced by legal summarizers. 
This promotes access to legal information and encourages a fair and just legal system. 

---

## Challenges addressed
- The pretrained model vocabulary is predominantly **US based. “US attorney office” pops up in Indian dataset summaries**
- The LLMs which can take in prompts cannot handle **token limits ie. the size of the legal documents**
- The extractive and other prior methods are **less human understandable**. (preferred by experts)

---
## Proposed system
![image](https://github.com/SwethaMagesh/sankshepika-mlpro/assets/43994542/e889f7fb-8e50-4542-87f2-7f5e14fd7670)
![image](https://github.com/SwethaMagesh/sankshepika-mlpro/assets/43994542/b5d372f2-c4b1-419a-80b1-2fc001b7e674)
![image](https://github.com/SwethaMagesh/sankshepika-mlpro/assets/43994542/1a1a9899-ada5-4981-88cb-8405425c10f5)



---
## Results
- Obtained ROUGE score of 0.45 on an average.
![image](https://github.com/SwethaMagesh/sankshepika-mlpro/assets/43994542/d9714f71-3df8-4592-89a6-e18b1230a991)

---
## Conclusion
- The pretrained models suffer from incorrect or hallucinated information especially from US Law
- Pre-trained transformer fine tuning takes a lot of time and resources
- Different approaches - Abstractive & Extractive have different pros and cons
- Combining summaries is easier for LLMs with token limit to get a reasonable output leveraging all methods


