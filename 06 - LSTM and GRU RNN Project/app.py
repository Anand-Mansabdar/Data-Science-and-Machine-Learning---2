import pickle as pkl
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("predict_word_lstm.h5")

with open("tokenizer.pkl", "rb") as file:
  tokenizer = pkl.load(file)
  
  

def predict_word(model, tokenizer, text, max_length):
  token_list = tokenizer.texts_to_sequences([text])[0]
  if len(token_list) >= max_length:
    token_list = token_list[-(max_length-1):]
  
  token_list = pad_sequences([token_list], maxlen=max_length-1, padding="pre")
  predicted = model.predict(token_list, verbose=0)
  predicted_word_idx = np.argmax(predicted, axis=1)
  for word, index in tokenizer.word_index.items():
    if index == predicted_word_idx:
      return word
  return None

st.title("Next Word Prediction")

input_text = st.text_input("Enter a sentence or a sequence of words:")
if st.button("Predict word"):
  max_length = model.input_shape[1]+1 
  st.write(f"Next word is {predict_word(model, tokenizer, input_text, max_length)}")
