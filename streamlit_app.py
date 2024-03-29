import streamlit as st
import pickle
import time 

# Load the model
# Load the model from a pickle file
with open('nlp.pkl', 'rb') as file:
    model = pickle.load(file)

#model = pickle.load('nlp.plk')
#model = joblib.load('text_clf_model.joblib')

#st.area_chart(df)
#st.bar_chart(df)
#st.line_chart(df)
#st.map(df)
#st.scatter_chart(df)

# Function to predict text and get score
def predict_text_with_score(model, text):
    prediction = model.predict([text])
    prediction_proba = model.predict_proba([text])
    class_labels = model.classes_
    proba_scores = {class_labels[i]: prediction_proba[0][i] for i in range(len(class_labels))}
    return prediction[0], proba_scores

# Streamlit user interface
st.markdown("<h1 style='text-align: center;'>Text Classification Application</h1>", unsafe_allow_html=True)

# Center the image by adjusting the weights of the columns
# Increase the weight of the center column if necessary
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.image("Thinking Brain.png", caption='Powered by Ai', width=250)
    
st.write("Make sure the text is not too short. Models often need a certain amount of context to make accurate predictions.")

col3.metric(label="Temperature", value="60 °C", delta="3 °C")
             
# User input text area
user_input = st.text_area("Enter text here:")
if st.button('Predict'):
        # Create a progress bar and fill it as the prediction is being made
    with st.empty():
        for percent_complete in range(100):
            time.sleep(0.005)  # Sleep for a short moment to simulate work being done
            st.progress(percent_complete + 1)
    prediction, scores = predict_text_with_score(model, user_input)
    st.write("Predicted class:", prediction)
    st.write("Probability Scores:", scores)

# Create a placeholder for the progress bar
#progress_bar_placeholder = st.empty()

# When the 'Predict' button is pressed
#if st.button('Predict'):

#progress_bar = col2.progress(0)
#for  perc_completed in range(100):
   # time.sleep(0.25)
    #progress_bar.progress(perc_completed+1)
    # Start the progress bar

    
#bar = st.progress(25)
