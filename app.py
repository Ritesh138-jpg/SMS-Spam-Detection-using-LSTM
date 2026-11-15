import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL=Path("models/sms_lstm.keras")
TOKENIZER=Path("models/tokenizer.joblib")
METRICS=Path("models/metrics.json")
MAX_LEN=100
st.set_page_config(page_title="SMS Spam Detection",page_icon="📩",layout="wide")
st.markdown("""<style>.block-container{max-width:1100px;padding-top:2rem}.hero{padding:1.5rem 1.8rem;border-radius:18px;background:linear-gradient(115deg,#172554,#312e81,#4338ca);color:white;margin-bottom:1.2rem}.hero h1{color:white;margin:0}.hero p{color:#dbeafe;margin:.4rem 0 0}</style>""",unsafe_allow_html=True)
@st.cache_resource
def load_model_files():
    if not MODEL.exists() or not TOKENIZER.exists(): return None,None
    return tf.keras.models.load_model(MODEL),joblib.load(TOKENIZER)
model,tokenizer=load_model_files()
st.markdown('<div class="hero"><h1>📩 SMS Spam Detection</h1><p>Detect suspicious messages using a Long Short-Term Memory neural network.</p></div>',unsafe_allow_html=True)
with st.sidebar:
    st.header("About this project")
    st.write("Model: LSTM")
    st.write("Task: Binary text classification")
    st.write("Classes: Spam / Ham")
    st.caption("Ham means a legitimate, non-spam message.")
    st.divider()
    st.caption("Predictions are model estimates, not a guarantee of safety.")
if model is None:
    st.error("Trained model files are missing. Run python train.py first.")
    st.stop()
metrics={}
if METRICS.exists():
    try: metrics=json.loads(METRICS.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError): pass
a,b,c,d=st.columns(4)
a.metric("Test accuracy",f"{metrics['accuracy']:.2%}" if "accuracy" in metrics else "—")
b.metric("Spam precision",f"{metrics['precision']:.2%}" if "precision" in metrics else "—")
c.metric("Spam recall",f"{metrics['recall']:.2%}" if "recall" in metrics else "—")
d.metric("Spam F1-score",f"{metrics['f1_score']:.2%}" if "f1_score" in metrics else "—")
st.subheader("Classify an SMS")
message=st.text_area("Enter a message",height=130,placeholder="Example: Congratulations! You have won a prize...")
if "history" not in st.session_state: st.session_state.history=[]
if st.button("🔍 Classify message",type="primary",use_container_width=True):
    if not message.strip(): st.warning("Please enter a message.")
    else:
        seq=tokenizer.texts_to_sequences([message.strip()])
        padded=pad_sequences(seq,maxlen=MAX_LEN,padding="post",truncating="post")
        spam_prob=float(model.predict(padded,verbose=0)[0][0])
        label="SPAM" if spam_prob>=0.5 else "HAM"
        confidence=spam_prob if label=="SPAM" else 1-spam_prob
        st.session_state.history.insert(0,{"message":message.strip(),"label":label,"confidence":confidence})
        st.session_state.history=st.session_state.history[:10]
        if label=="SPAM": st.error("🚨 SPAM DETECTED")
        else: st.success("✅ HAM — LIKELY LEGITIMATE")
        st.progress(confidence,text=f"Confidence for predicted class: {confidence:.1%}")
        st.caption(f"Estimated spam probability: {spam_prob:.1%}")
st.divider()
left,right=st.columns([1.2,0.8])
with left:
    st.subheader("Recent predictions")
    if st.session_state.history:
        hist=pd.DataFrame([{"Message":x["message"][:65]+("…" if len(x["message"])>65 else ""),"Prediction":x["label"],"Confidence":f"{x['confidence']:.1%}"} for x in st.session_state.history])
        st.dataframe(hist,use_container_width=True,hide_index=True)
        if st.button("Clear history"): st.session_state.history=[]; st.rerun()
    else: st.info("Your recent predictions will appear here during this session.")
with right:
    st.subheader("Dataset overview")
    st.metric("Total messages","5,574")
    st.metric("Legitimate (ham)","4,827 · 86.6%")
    st.metric("Spam","747 · 13.4%")
    st.caption("Source: UCI SMS Spam Collection.")
if metrics.get("confusion_matrix"):
    st.divider()
    st.subheader("Test-set confusion matrix")
    st.dataframe(pd.DataFrame(metrics["confusion_matrix"],index=["Actual ham","Actual spam"],columns=["Predicted ham","Predicted spam"]),use_container_width=True)
    st.caption(f"Held-out test messages: {metrics.get('test_samples','—')}")
st.caption("Academic prototype • LSTM text classifier • UCI SMS Spam Collection")