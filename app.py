import streamlit as st
import tensorflow as tf
import numpy as np
import pickle


# Custom perplexity metric; expand y_true so that its rank matches y_pred.
def perplexity(y_true, y_pred):
    y_true_exp = tf.expand_dims(y_true, axis=-1)
    cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(y_true_exp, y_pred)
    return tf.exp(tf.reduce_mean(cross_entropy))


# Load the saved tokenizer (ensure you have saved this using pickle during training)
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the saved model
model = tf.keras.models.load_model('model.keras', custom_objects={'perplexity': perplexity})

# Use the same max sequence length as during training
max_seq_len = 100  # Adjust if you used a different value during training

def generate_text(seed_word, next_words, max_seq_len, temperature=0.8):
    """
    Generates two related verses from a seed word using temperature-based sampling.
    """
    def generate_verse(seed_text, next_words, max_seq_len, temperature):
        for _ in range(next_words):
            # Tokenize and pad the seed text
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = tf.keras.preprocessing.sequence.pad_sequences(
                [token_list], maxlen=max_seq_len - 1, padding='pre')
            # Predict next word probabilities
            predicted_probs = model.predict(token_list, verbose=0)[0]
            # Apply temperature-based sampling
            predicted_probs = np.log(predicted_probs + 1e-8) / temperature
            predicted_probs = np.exp(predicted_probs) / np.sum(np.exp(predicted_probs))
            # Sample the next word using the adjusted probability distribution
            predicted_index = np.random.choice(range(len(predicted_probs)), p=predicted_probs)
            output_word = tokenizer.index_word.get(predicted_index, "")
            seed_text += " " + output_word
        return seed_text

    # Generate the first verse from the seed word
    verse1 = generate_verse(seed_word, next_words, max_seq_len, temperature)
    # Use the last 3 words of the first verse as a seed for the second verse
    last_words = " ".join(verse1.split()[-3:])
    verse2 = generate_verse(last_words, next_words, max_seq_len, temperature)
    return verse1, verse2

# Streamlit UI
st.title("Roman Urdu Poetry Generator")
st.write("Enter a seed text and generate two verses of poetry:")

seed_text = st.text_input("Seed Text", "Ishq")
next_words = st.slider("Number of words to generate per verse", min_value=5, max_value=50, value=20)
temperature = st.slider("Temperature", min_value=0.1, max_value=1.5, value=0.8)

if st.button("Generate Poetry"):
    verse1, verse2 = generate_text(seed_text, next_words, max_seq_len, temperature)
    st.markdown("### Verse 1")
    st.write(verse1)
    st.markdown("### Verse 2")
    st.write(verse2)
