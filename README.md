# ner-movie
Named Entity Recognition model on movie domain

## Output
'2017년에 개봉한 미국 스릴러 영화 알려줘' => 

    {'actor': [], 'country': ['미국'], 'director': [], 'genre': ['스릴러'], 
                          'month': [], 'movie': [], 'year': ['2017년']}
## Motivation for making this model
    C'est La Vie team of AI-LAB KOREA(한국인공지능연구소) worked on a chatbot project in 2018.
    This model was made by me and it was used for the chatbot project.
    
## How to train the ner-movie model
    python main.py
    
## Model Structure
    - use five word embeddings : 
      (1) pretrained wiki-corpus word2vec embedding
      (2) pretrained movie domain word2vec embedding
      (3) POS(Part of Speech) embedding
      (4) syllable(char) embedding
      (5) entity dictionary embedding 
    - bi-directional lstm 
    - softmax 
## Data on "data" folder
    - intent_v3.txt: input data(anticipated user's questions about movies on a chatbot system)

## Other Things to Know
    - labels: in order to save time, label datas are not manually made. 
              I made codes to generate labels using entity dictionaries. 
## Reference
    - 송치윤 외, 「개선된 워드 임베딩 모델과 사전을 이용한 Bidrectional LSTM CRF 기반의 한국어 개체명 」, 한국소프트웨어종합학술대회 논문집, 2017
      => I obtained an idea of concatenating five different types of word embeddings
      
