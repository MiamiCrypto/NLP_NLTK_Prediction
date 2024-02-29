import streamlit as st
import pickle

# Load the model
# Load the model from a pickle file
with open('nlp.pkl', 'rb') as file:
    model = pickle.load(file)

#model = pickle.load('nlp.plk')
#model = joblib.load('text_clf_model.joblib')

# Function to predict text and get score
def predict_text_with_score(model, text):
    prediction = model.predict([text])
    prediction_proba = model.predict_proba([text])
    class_labels = model.classes_
    proba_scores = {class_labels[i]: prediction_proba[0][i] for i in range(len(class_labels))}
    return prediction[0], proba_scores

# Streamlit user interface
st.markdown("<h1 style='text-align: center;'>Text Prediction App</h1>", unsafe_allow_html=True)
#st.title('Text Prediction App')
col1, col2, col3 = st.beta_columns([1,1,1])
with col2:
    st.image("Thinking Brain.png", caption='Flight by machines heavier than air is unpractical and insignificant, if not utterly impossible.', width=250)
user_input = st.text_area("Enter text here:")
if st.button('Predict'):
    prediction, scores = predict_text_with_score(model, user_input)
    st.write("Predicted class:", prediction)
    st.write("Probability Scores:", scores)
