import pickle
import streamlit as st
import nltk
import string
from nltk.stem import PorterStemmer
nltk.download('punkt')

st.set_page_config(layout='wide')
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load((open('model.pkl', 'rb')))
new_one = open("stop_items.txt", "r")
data = new_one.read()
stop_words = data.split("\n")
ps = PorterStemmer()


def empty_space():
    col4, col5, col6 = st.columns(3)
    with col4:
        st.write('')
        st.write('')
        st.write('')
    with col5:
        pass
    with col6:
        pass


def transform(obj):
    obj = obj.lower()
    obj = nltk.word_tokenize(obj)

    x = []
    for i in obj:
        if i.isalnum():
            x.append(i)

    y = x.copy()
    x.clear()

    for i in x:
        if i not in stop_words and i not in string.punctuation:
            y.append(ps.stem(i))

    return ' '.join(y)


st.title('Email Spam Detection')

empty_space()

inp = st.text_input('Enter the Message / Email')


empty_space()

btn = st.button('Predict')

empty_space()

if btn:
    transformed_inp = transform(inp)
    data = tfidf.transform([transformed_inp])
    result = model.predict(data)[0]

    if result == 1:
        st.subheader('Spam')
    else:
        st.subheader('Not Spam')
