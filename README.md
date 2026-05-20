#  AI Document Analyzer & Interactive QA Assistant

Un sistem avansat de procesare a limbajului natural (NLP) dezvoltat în Python, capabil să analizeze complet documente textuale utilizând **5 modele de Inteligență Artificială (arhitecturi Transformer)** distincte de pe HuggingFace.

##  Funcționalități
* **Statistici text:** Curățare automată prin Regex și extragere manuală a cuvintelor cheie (Term Frequency).
* **Language Detection:** Identificarea automată a limbii (suportă 51 de limbi).
* **Named Entity Recognition (NER):** Extragerea automată de persoane, locații și organizații folosind formatul BIO pe modelul BERT.
* **Zero-Shot Classification & Style Analysis:** Clasificarea tematică și determinarea stilului/tonului prin inferență logică (NLI) cu modelul BART.
* **Abstractive Summarization:** Generarea unui rezumat inteligent prin mecanismul *Beam Search* (`num_beams=4`).
* **Interactive Question Answering:** Un asistent virtual extractiv bazat pe modelul RoBERTa (antrenat pe SQuAD 2.0) care răspunde la întrebări live din text.
