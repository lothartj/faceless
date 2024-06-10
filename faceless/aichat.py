import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load initial dataset
initial_data = {
    'questions': ["What is your name?", "How are you?", "Tell me a joke."],
    'answers': ["I am a chatbot.", "I'm doing well, thank you!", "Why don't scientists trust atoms? Because they make up everything."]
}

df = pd.DataFrame(initial_data)

# Initialize global variables
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['questions'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to update the dataset and retrain the model
def update_model(question, answer):
    global df, vectorizer, cosine_sim
    df = df.append({'questions': question, 'answers': answer}, ignore_index=True)
    
    # Update the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['questions'])

    # Update the cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return vectorizer, cosine_sim

# Function to get a response from the chatbot
def get_response(user_input):
    user_input_tfidf = vectorizer.transform([user_input])
    similarity_scores = cosine_sim.dot(user_input_tfidf.T).flatten()
    
    # Find all indices with the maximum value
    max_indices = np.where(similarity_scores == similarity_scores.max())[0]
    
    # Randomly select one of the indices if there are multiple with the maximum value
    most_similar_index = np.random.choice(max_indices)
    
    return df['answers'][most_similar_index]

# Streamlit app
def main():
    global vectorizer, cosine_sim
    st.title("Self-Learning Chatbot")

    user_input = st.text_input("You: ")

    if st.button("Send"):
        st.text("Chatbot: " + get_response(user_input))

        user_feedback = st.radio("Was this response helpful?", ("Yes", "No"))
        
        if user_feedback == "Yes":
            new_question = user_input
            new_answer = st.text_input("Provide a better response: ")
            if st.button("Submit Feedback"):
                vectorizer, cosine_sim = update_model(new_question, new_answer)
                st.success("Feedback received and model updated!")

if __name__ == "__main__":
    # Run Streamlit app
    main()
