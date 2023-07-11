import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from wordcloud import WordCloud
import nltk


nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

# Load the trained model
with open('./model/naive_bayes_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the TfidfVectorizer
with open('./model/tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Function to preprocess the input text
def preprocess_text(text):
    if text is not None and isinstance(text, str):
        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # Remove usernames starting with '@'
        text = re.sub(r'@\w+', '', text)

        # Remove hashtags starting with '#'
        text = re.sub(r'#\w+', '', text)

        # Remove non-alphabetic characters
        text = re.sub('[^a-zA-Z]', ' ', text)

        # Lemmatize words
        lemmatizer = WordNetLemmatizer()
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

        # Remove words with 3 characters or less
        text = ' '.join([word for word in text.split() if len(word) > 3])

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in stop_words])

        # Remove duplicate words while preserving the order
        words = text.split()
        text = ' '.join(list(dict.fromkeys(words)))

    else:
        text = ''
    return text

# Function to predict sentiment
def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)
    features = vectorizer.transform([preprocessed_text])
    prediction = model.predict(features)[0]
    return prediction

# Sentiment Detection Tab
def sentiment_detection():
    st.title("Sentiment Detection")
    sentence = st.text_input("Enter a sentence:")
    if st.button("Predict"):
        if sentence:
            sentiment = predict_sentiment(sentence)
            st.write("Sentiment:", sentiment)
        else:
            st.write("Please enter a sentence.")


# Define a function to get the sentiment polarity score
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Define a function to categorize the sentiment polarity score into 4 categories
def categorize_sentiment(score):
    if score >= 0.5:
        return 'Positive'
    elif score >= 0.05 and score < 0.5:
        return 'Moderately Positive'
    elif score > -0.05 and score < 0.05:
        return 'Neutral'
    elif score > -0.5 and score <= -0.05:
        return 'Moderately Negative'
    else:
        return 'Negative'





# Function to generate and display word cloud
def generate_wordcloud(df, sentiment):
    # Filter the data for the specified sentiment
    filtered_data = df[df['sentiment_textblob'] == sentiment]

    # Concatenate the text data into a single string
    words = ' '.join(filtered_data['clean_text'].values)

    # Create the WordCloud object
    wordcloud = WordCloud(background_color='white', width=800, height=400).generate(words)

    # Display the word cloud in Streamlit
    st.write(f"Word Cloud - {sentiment} Sentiment")
    st.image(wordcloud.to_array())



    
# Data Visualization Tab
def exploratory_data_analysis():
    st.title("Exploratory Data Analysis")
    # Assuming you have a DataFrame 'df' containing the sentiment data
    df = pd.read_csv('./data/Tweets.csv')
    st.write("Load Data:")
    st.dataframe(df.sample(10))

    st.write("Data Cleaning")
    df['clean_text'] = df['text'].apply(preprocess_text)
    sample_df = df.sample(10)[['text', 'clean_text']]
    st.dataframe(sample_df)
    
    st.write("Sentiment Analysis")
    df['textblob_polarity'] = df['clean_text'].apply(get_sentiment).round(2)
    df['sentiment_textblob'] = df['textblob_polarity'].apply(categorize_sentiment)
    df = df[['clean_text', 'textblob_polarity', 'sentiment_textblob']]
    st.dataframe(df)
    
    # Count the sentiment labels
    sentiment_counts = df['sentiment_textblob'].value_counts()

    # Plot the sentiment counts using Matplotlib
    st.write("Sentiment Barchart")
    st.bar_chart(sentiment_counts)
    plt.ylabel("Label")
    plt.title("Label Counts")
    sentiment_counts.plot(kind="bar")
    plt.show()  


    # Display the sentiment counts using Streamlit
    st.write("Sentiment Label Counts")
    st.dataframe(sentiment_counts)
    
    # Generate and display word clouds for positive and negative sentiments
    generate_wordcloud(df, 'Positive')
    generate_wordcloud(df, 'Negative')
    
    
    


    







# About Me Tab
def about_me():
    st.title("About Me")
    st.write("Hey everyone! I'm Christian M. De Los Santos, from the Philippines. I have over 2 years of experience in the field of data analytics, with a special focus on machine learning. I firmly believe that AI and ML have the power to bring about positive change in our communities, which is why I'm here, eager to make an impact. Learning from all of you brilliant minds is something I'm truly looking forward to. Let's collaborate and create something amazing together!")
    st.write("Contact: christiandelossantos444@gmail.com")
    
    # # Social Media Icons
    # st.markdown("""
    #     <style>
    #         .social-media-icons {
    #             display: flex;
    #             margin-top: 20px;
    #         }
    #         .social-media-icons a {
    #             margin: 0 10px;
    #         }
    #     </style>
    #     <div class="social-media-icons">
    #         <a href="https://twitter.com/your_twitter_account"><img src="https://example.com/twitter-icon.png" alt="Twitter" width="30"></a>
    #         <a href="https://linkedin.com/in/your_linkedin_account"><img src="https://example.com/linkedin-icon.png" alt="LinkedIn" width="30"></a>
    #         <a href="https://github.com/your_github_account"><img src="https://example.com/github-icon.png" alt="GitHub" width="30"></a>
    #     </div>
    # """, unsafe_allow_html=True)

# Main Streamlit app
def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Go to", ["Sentiment Detection", "Exploratory Data Analysis", "About Me"])

    if app_mode == "Sentiment Detection":
        sentiment_detection()
    elif app_mode == "Exploratory Data Analysis":
        exploratory_data_analysis()
    elif app_mode == "About Me":
        about_me()

if __name__ == '__main__':
    main()
