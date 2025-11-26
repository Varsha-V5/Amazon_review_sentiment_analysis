import streamlit as st
import numpy as np
import pickle
import re
import string
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk import word_tokenize

# --- Preprocessing Functions (MUST match training script exactly) ---

def clean_text(text):
    """
    Replicates the exact cleaning function used during model training.
    """
    text = re.sub('\[.*?\]', '', text) #remove brackets and words
    text = re.sub('https?://\S+|www\.\S+', '', text) #remove urls
    text = re.sub('<.*?>+', '', text) #remove html tags
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)# remove punctuationz
    text = re.sub('\n', '', text) #remove newline characters from the text.
    text = re.sub('\w*\d\w*', '', text) #remove digits and words containing digits (e.g., 'item4')
    text = re.sub(r'[^\x00-\x7F]+', '', text) # remove emoji
    text = text.lower()
    return text

# --- Load Artifacts ---

@st.cache_resource
def load_artifacts():
    """Load the model and tokenizer once for the Streamlit session."""
    try:
        # Load saved tokenizer
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)

        # Load trained BiLSTM model
        model = tf.keras.models.load_model("sentiment_bilstm_model.h5")
        return model, tokenizer
    except FileNotFoundError:
        st.error("Error: Model or Tokenizer files not found. Please ensure 'tokenizer.pkl' and 'sentiment_bilstm_model.h5' are in the same directory.")
        return None, None
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None

model, tokenizer = load_artifacts()

# --- Streamlit UI and Prediction Logic ---

st.set_page_config(page_title="BiLSTM Sentiment Analyzer", layout="centered")

st.title("Amazon Review Sentiment Classifier")
st.markdown("---")
st.subheader("BiLSTM Model Deployment")

if model and tokenizer:
    
    user_input = st.text_area(
        "Enter Review Text:",
        placeholder="e.g., This product is the best value for money, highly recommend!",
        height=150
    )
    
    maxlen = 200
    label_map = {
        0: "Negative (Rating 1-2)", 
        1: "Neutral (Rating 3)", 
        2: "Positive (Rating 4-5)"
    }

    if st.button("Predict Sentiment", type="primary"):
        if user_input.strip() == "":
            st.warning("Please enter a review to analyze.")
        else:
            with st.spinner('Analyzing review...'):
                
                # 1. Clean Text
                cleaned_text = clean_text(user_input)
                
                # 2. Tokenize (matches the structure used before fitting the tokenizer)
                # The tokenizer was fit on a series of lists of tokens
                token_list = word_tokenize(cleaned_text)
                
                # 3. Convert tokens to sequences
                seq = tokenizer.texts_to_sequences([token_list])
                
                # 4. Pad Sequence
                pad = pad_sequences(seq, maxlen=maxlen, padding='post', truncating='post')
                
                # 5. Predict
                pred_probs = model.predict(pad)[0]
                pred_label = np.argmax(pred_probs)

                st.markdown("### Prediction Result")

                predicted_sentiment = label_map[pred_label]
                if pred_label == 2:
                    st.success(f"**Predicted Sentiment:** {predicted_sentiment}")
                elif pred_label == 0:
                    st.error(f"**Predicted Sentiment:** {predicted_sentiment}")
                else:
                    st.info(f"**Predicted Sentiment:** {predicted_sentiment}")

                st.markdown("### Confidence Scores")
                
                # Format scores for display
                scores_display = {label_map[i]: f"{float(pred_probs[i]):.4f}" for i in range(3)}
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Negative", scores_display[label_map[0]])
                col2.metric("Neutral", scores_display[label_map[1]])
                col3.metric("Positive", scores_display[label_map[2]])

    st.markdown("---")
    st.caption("Note: Prediction accuracy for negative and neutral reviews may be low due to aggressive text cleaning during the original model training.")