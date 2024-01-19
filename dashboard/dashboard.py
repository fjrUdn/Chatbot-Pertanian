import streamlit as st
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import time
from transformers import AutoTokenizer,AdamW,BertForQuestionAnswering, AutoModel

tokenizer = AutoTokenizer.from_pretrained("Rifky/Indobert-QA")

# Load the fine-tuned modeol
model = torch.load("dashboard/model3_1.9",map_location=torch.device('cpu'))
#model = AutoModel.from_pretrained("Rifky/Indobert-QA")
model.eval()

#fungsi prediksi

def predict(context,query):

  inputs = tokenizer.encode_plus(query, context, return_tensors='pt')

  outputs = model(**inputs)
  answer_start = torch.argmax(outputs[0])  # get the most likely beginning of answer with the argmax of the score
  answer_end = torch.argmax(outputs[1]) + 1

  answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

  return answer

def normalize_text(s):
  """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
  import string, re

  def remove_articles(text):
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)

  def white_space_fix(text):
    return " ".join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction, truth):
  pred_tokens = normalize_text(prediction).split()
  truth_tokens = normalize_text(truth).split()

  # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
  if len(pred_tokens) == 0 or len(truth_tokens) == 0:
    return int(pred_tokens == truth_tokens)

  common_tokens = set(pred_tokens) & set(truth_tokens)

  # if there are no common tokens then f1 = 0
  if len(common_tokens) == 0:
    return 0

  prec = len(common_tokens) / len(pred_tokens)
  rec = len(common_tokens) / len(truth_tokens)

  return 2 * (prec * rec) / (prec + rec)

def compute_precision(prediction, truth):
  pred_tokens = normalize_text(prediction).split()
  truth_tokens = normalize_text(truth).split()

  # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
  if len(pred_tokens) == 0 or len(truth_tokens) == 0:
    return int(pred_tokens == truth_tokens)

  common_tokens = set(pred_tokens) & set(truth_tokens)

  # if there are no common tokens then f1 = 0
  if len(common_tokens) == 0:
    return 0

  prec = len(common_tokens) / len(pred_tokens)
  rec = len(common_tokens) / len(truth_tokens)

  return prec

def compute_recall(prediction, truth):
  pred_tokens = normalize_text(prediction).split()
  truth_tokens = normalize_text(truth).split()

  # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
  if len(pred_tokens) == 0 or len(truth_tokens) == 0:
    return int(pred_tokens == truth_tokens)

  common_tokens = set(pred_tokens) & set(truth_tokens)

  # if there are no common tokens then f1 = 0
  if len(common_tokens) == 0:
    return 0

  prec = len(common_tokens) / len(pred_tokens)
  rec = len(common_tokens) / len(truth_tokens)

  return rec

def give_an_answer(context,query,answer):
  prediction = predict(context,query)
  em_score = compute_exact_match(prediction, answer)
  f1_score = compute_f1(prediction, answer)
  prec = compute_precision(prediction, answer)
  rec = compute_recall(prediction, answer)

  st.write(f"Question: {query}")
  st.write(f"Prediction: {prediction}")
  st.write(f"True Answer: {answer}")
  st.write(f"EM: {em_score}")
  st.write(f"F1: {f1_score}")
  st.write(f"Precision: {prec}")
  st.write(f"Recall: {rec}")
  st.write("\n")

context = "Hama dan penyakit tumbuhan merupakan jenis organisme pengganggu tumbuhan (OPT), selain gulma. Serangan hama dan penyakit pada tanaman dapat menyebabkan kerugian besar pada tanaman dan dapat mengancam perekonomian petani. Penyebaran hama dan penyakit tanaman meningkat drastis dalam beberapa tahun terakhir. Hama dan penyakit tanaman mudah menyebar ke beberapa negara dan mencapai proporsi epidemi. Belalang, lalat buah, ulat grayak, penyakit antaknose, fuso, penyakit virus kerdil, busuk buah adalah beberapa hama dan penyakit tanaman yang paling merusak. Tiga cara penyebaran hama dan penyakit tanaman yaitu dengan cara: 1) perdagangan atau migrasi 2) pengaruh lingkungan, seperti faktor cuaca, angin, percikan air hujan, dan 3) faktor biotik berupa: serangga atau vektor lainnya."

with st.sidebar:
  st.image('bg.png')
  st.radio(
      "Topik",
      key="visibility",
      options=["hama", "solusi", "knowledge"],
  )

st.title('Chatbot Pertanian')

# for q,a in zip(queries,answers):
#   give_an_answer(context,q,a)
# tanya = "setiap perbuatan manusia tergantung dari apa?"
st.write(
  ''' 
  Dengan menggunakan model Indobert, chatbot ini tidak hanya mampu memahami pertanyaan dan permintaan petani dengan akurat, 
  tetapi juga memberikan jawaban yang relevan dan dapat dipahami.
  '''
  )
        # '''
        # Kemampuan chatbot untuk memproses bahasa alami memungkinkan 
        #  pengguna berinteraksi dengan aplikasi secara intuitif, menjadikan pengalaman pengguna lebih mudah dan efektif. 
        #  Selain itu, chatbot juga memiliki kemampuan untuk mempelajari pola perilaku pengguna dari waktu ke waktu, mempersonalisasi rekomendasi, 
        #  dan menyediakan solusi yang lebih spesifik sesuai dengan kebutuhan individual petani. Dengan adanya chatbot ini, 
        #  akses petani terhadap informasi pertanian terkini dan solusi praktis dapat ditingkatkan secara signifikan, mempercepat proses 
        #  pengambilan keputusan dan meningkatkan kualitas pertanian secara keseluruhan.
        #  '''
tanya = st.text_input('Silahkan berikan pertanyaan anda', '')
prediction2 = predict(context,tanya)
st.write(f"Prediction: {prediction2}")