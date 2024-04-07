import pandas as pd
import pickle
from flask import Flask, render_template,request
import prep
app = Flask(__name__)
x = pickle.load(open('test.pkl', 'rb'))
data = pd.read_excel("main.xlsx")
que = x.m()
qt = x.l()
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    lis=data['topic']
    w=list(request.form.values())[0]
    indi=[]
    st=str(w)
    i=0
    for t in lis:
        if x.search_word(str(t),st):
            indi.append(i)
        i=i+1
    indi=indi[1:5]
    # print(str(st))
    print(indi)
    questions=[]
    for i in indi:
        qs=[]    
        text=data['Text'][i]
        summarized_text = x.summarizer(text,x.summary_model,x.summary_tokenizer)
        # print(summarized_text)
        imp_keywords = x.get_keywords(text,summarized_text)
        # print(imp_keywords)
        for answer in imp_keywords:
            ques = x.get_question(summarized_text,answer,que,qt)
            ques=ques+"__ANS:"+answer
            qs.append(ques)
        qsz=set(qs)
        for z in qsz:
            questions.append(z)
    return render_template("index.html",prediction_text=questions)


# Add Flask routes or functions to handle your desired operations
if __name__ == "__main__":
    app.run(debug=True, host='localhost', port=5500)

