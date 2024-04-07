import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
import nltk
# nltk.download('punkt')
# nltk.download('brown')
# nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
# import pandas as pd
# nltk.download('stopwords')
from nltk.corpus import stopwords
import string
import traceback
from flashtext import KeywordProcessor
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
import string
# import pke
import traceback
import spacy
nlp = spacy.load("en_core_web_sm")
import en_core_web_sm
nlp = en_core_web_sm.load()
# from rake_nltk import Rake
class prep:
    # data=pd.read_e("main.xlsx")
    def search_word(self,sentence, word):
        sentence_lower = sentence.lower()
        word_lower = word.lower()
        if word_lower in sentence_lower:
            return True
        else:
            return False
    summary_model = T5ForConditionalGeneration.from_pretrained('t5-base')
    summary_tokenizer = T5Tokenizer.from_pretrained('t5-base', model_max_length=1024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary_model = summary_model.to(device)
    
    def postprocesstext(self,content):
        final=""
        for sent in sent_tokenize(content):
            sent = sent.capitalize()
            final = final +" "+sent
        return final
    
    def summarizer(self,text,model,tokenizer):
        text = text.strip().replace("\n"," ")
        text = "summarize: "+text
        # print (text)
        # max_len = 512
        encoding = tokenizer.encode_plus(text,max_length=1024, pad_to_max_length=False,truncation=True, return_tensors="pt")

        input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

        outs = model.generate(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        early_stopping=True,
                                        num_beams=3,
                                        num_return_sequences=1,
                                        no_repeat_ngram_size=2,
                                        min_length = 75,
                                        max_length=300)
        dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]
        summary = dec[0]
        summary = self.postprocesstext(summary)
        summary= summary.strip()
        return summary
    
    def get_nouns_multipartite(self,content):
        out=[]
        try:
            extractor = pke.unsupervised.MultipartiteRank()
            extractor.load_document(input=content,language='en',spacy_model=nlp)
            #    not contain punctuation marks or stopwords as candidates.
            pos = {'PROPN','NOUN'}
            #pos = {'PROPN','NOUN'}
            stoplist = list(string.punctuation)
            stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
            stoplist += stopwords.words('english')
            # extractor.candidate_selection(pos=pos, stoplist=stoplist)
            extractor.candidate_selection(pos=pos)
            # 4. build the Multipartite graph and rank candidates using random walk,
            #    alpha controls the weight adjustment mechanism, see TopicRank for
            #    threshold/method parameters.
            extractor.candidate_weighting(alpha=1.1,
                                        threshold=0.75,
                                        method='average')
            keyphrases = extractor.get_n_best(n=15)
            for val in keyphrases:
                out.append(val[0])
        except:
            out = []
            traceback.print_exc()
        return out

    def get_keywords(self,originaltext,summarytext):
        keywords = self.get_nouns_multipartite(originaltext)
        print ("keywords unsummarized: ",keywords)
        keyword_processor = KeywordProcessor()
        for keyword in keywords:
            keyword_processor.add_keyword(keyword)

        keywords_found = keyword_processor.extract_keywords(summarytext)
        keywords_found = list(set(keywords_found))
        print ("keywords_found in summarized: ",keywords_found)

        important_keywords =[]
        for keyword in keywords:
            if keyword in keywords_found:
                important_keywords.append(keyword)

        return important_keywords[:4]
    # imp_keywords = get_keywords(text,summarized_text)
    # print (imp_keywords)
#     def get_keywords(self,text):
# #     text='Natural Language Generation (NLG) is a branch of artificial intelligence (AI) and natural language processing (NLP) that focuses on generating human-like text or speech based on given input or data. NLG algorithms aim to produce coherent, contextually relevant, and linguistically appropriate output that resembles human-generated language.NLG systems utilize various techniques and approaches to generate text. These may include rule-based systems, statistical models, or more advanced deep learning architectures such as recurrent neural networks (RNNs) and transformer models like GPT (Generative Pre-trained Transformer). These models learn from large amounts of text data to capture patterns, semantics, and stylistic nuances in language.'
#         r=Rake(punctuations=[')','(',',','.',':',').','),'])
#         r.extract_keywords_from_text(text)
#     #     print(r.get_ranked_phrases_with_scores())
#         implist=[]
#         for rating,keyword in r.get_ranked_phrases_with_scores():
#             if rating>=4:
#                 implist.append(keyword)
#         return implist[:4] 
    def m(self):
        question_model = T5ForConditionalGeneration.from_pretrained('Koundinya-Atchyutuni/t5-end2end-questions-generation')
#         question_model = question_model
        return question_model
    def l(self):
        question_tokenizer = T5Tokenizer.from_pretrained('Koundinya-Atchyutuni/t5-end2end-questions-generation')
        return question_tokenizer
    
    def get_question(self,context,answer,model,tokenizer):
        text = "context: {} answer: {}".format(context,answer)
        encoding = tokenizer.encode_plus(text,max_length=384, pad_to_max_length=False,truncation=True, return_tensors="pt")
        input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
        outs = model.generate(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        early_stopping=True,
                                        num_beams=5,
                                        num_return_sequences=1,
                                        no_repeat_ngram_size=2,
                                        max_length=72)
        dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]
        Question = dec[0].replace("question:","")
        Question= Question.strip()
        return Question
import pickle
obj=prep()
pickle.dump(obj,open('test.pkl','wb'))    



