from random import choice
import streamlit as st
from time import sleep
from stqdm import stqdm  # for getting animation after submit event
import json
import spacy
import spacy_streamlit
from transformers import pipeline
import os


#
# def draw_all():
#     st.markdown(
#         """
#         <style>
#         .title-text {
#             font-size: 36px;
#             font-weight: bold;
#             color: #4CAF50;
#             text-align: center;
#         }
#         .sub-text {
#             font-size: 16px;
#             color: #555;
#             text-align: center;
#             margin-bottom: 20px;
#         }
#         .features-list {
#             margin: 0 auto;
#             padding: 0;
#             max-width: 500px;
#             font-size: 18px;
#         }
#         .features-list li {
#             padding: 5px 0;
#         }
#         </style>
#
#         <div class="title-text">NLP Web App</div>
#         <p class="sub-text">
#             Unlock the power of Natural Language Processing with this web app, capable of handling anything you imagine with text. 🚀
#         </p>
#         <ul class="features-list">
#             <li>✨ Advanced Text Summarizer</li>
#             <li>📍 Named Entity Recognition</li>
#             <li>😊 Sentiment Analysis</li>
#             <li>❓ Question Answering</li>
#             <li>📝 Text Completion</li>
#         </ul>
#         """,
#         unsafe_allow_html=True,
#     )

def draw_all():
    st.markdown(
        """
        <style>
        /* General Page Styling */
        body {
            background-color: #f7f9fc;
            font-family: 'Arial', sans-serif;
        }

        /* Title Styling */
        .title-text {
            font-size: 40px;
            font-weight: bold;
            color: rgb(250 250 250);
            text-align: center;
            margin-bottom: 10px;
        }

        /* Subtitle Styling */
        .sub-text {
            font-size: 18px;
            color: rgb(230 234 241 / 65%);
            text-align: center;
            margin-bottom: 6px;
        }

        /* Features List Styling */
        .features-container {
            display: flex;
            justify-content: center;
        }
        .features-list {
            margin: 0;
            padding: 11px 28px;
            max-width: 600px;
            background: linear-gradient(135deg,#2e8cc4,#ff8e61);

            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            font-size: 20px;
            color: #ffffff;
            list-style-type: none;
            line-height: 1.5;
        }
        .features-list li {
            position: relative;
            padding: 10px 0;
        }
        .features-list li:before {
            content: '';
            margin-right: 10px;
            color: #264653;
        }
        </style>

        <div class="title-text">Welcome to NLP Genie </div>
        <p class="sub-text">
            Unleash NLP Genie! 🚀 Perfect for researchers,<br>
            businesses, and creators—turn complex texts into actionable insights instantly!
        </p>
        <div class="features-container">
            <ul class="features-list">
                <li>✨Smart Text Summarizer</li>
                <li>📍Named Entity Recognition</li>
                <li>💬Sentiment Analysis</li>
                <li>❓Question Answering</li>
                <li>📝Text Completion</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    draw_all()


def main():
    st.title("NLP Genie")
    menu = ["--Select--", "Summarizer", "Named Entity Recognition",
            "Sentiment Analysis", "Question Answering", "Text Completion"]
    choice = st.sidebar.selectbox("Choose what you want to do!", menu)

    if choice == "--Select--":
        st.write("""
                 This is a Natural Language Processing-Based Web App that can do   
                 anything you can imagine with text.
        """)

        st.image('banner_image.jpg')

    elif choice == "Summarizer":
        st.subheader("Text Summarization")
        st.write("Enter the text you want to summarize!")
        raw_text = st.text_area("Your Text", "")
        num_words = st.number_input("Enter the minimum number of words in the summary", min_value=10, step=1)

        if raw_text and num_words:
            summarizer = pipeline('summarization')
            summary = summarizer(raw_text, min_length=int(num_words), max_length=50)
            result_summary = summary[0]['summary_text']
            result_summary = '. '.join(list(map(lambda x: x.strip().capitalize(), result_summary.split('.'))))
            st.write(f"Here's your Summary:\n{result_summary}")

    elif choice == "Named Entity Recognition":
        nlp = spacy.load("en_core_web_sm")
        st.subheader("Named Entity Recognition")
        st.write("Enter the text below to extract named entities!")
        raw_text = st.text_area("Your Text", "Enter text here")

        if raw_text and raw_text != "":
            doc = nlp(raw_text)
            for _ in stqdm(range(50), desc="Please wait a bit. The model is fetching the results!"):
                sleep(0.1)
            spacy_streamlit.visualize_ner(doc, labels=nlp.get_pipe("ner").labels, title="List of Entities")

    elif choice == "Sentiment Analysis":
        st.subheader("Sentiment Analysis")
        sentiment_analysis = pipeline("sentiment-analysis")
        st.write("Enter the text below to find its sentiment!")
        raw_text = st.text_area("Your Text", "")

        if raw_text and raw_text != "":
            result = sentiment_analysis(raw_text)[0]
            sentiment = result['label']
            for _ in stqdm(range(50), desc="Please wait a bit. The model is fetching the results!"):
                sleep(0.1)
            if sentiment == "POSITIVE":
                st.write("# This text has a Positive Sentiment. 🤗")
            elif sentiment == "NEGATIVE":
                st.write("# This text has a Negative Sentiment. 😤")
            else:
                st.write("# This text seems Neutral ... 😐")

    elif choice == "Question Answering":
        st.subheader("Question Answering")
        st.write("Enter the context and ask a question to find the answer!")
        question_answering = pipeline("question-answering")

        context = st.text_area("Context", "")
        question = st.text_area("Your Question", "")

        if context and question and context != "Enter context here" and question != "Enter your question here":
            result = question_answering(question=question, context=context)
            answer = result['answer']
            st.write(f"Here's your Answer:\n{answer}")

    elif choice == "Text Completion":
        st.subheader("Text Completion")
        st.write("Enter the incomplete text to complete it automatically using AI!")
        text_generation = pipeline("text-generation")
        message = st.text_area("Your Text", "")

        if message and message != "Enter the text to complete":
            generator = text_generation(message)
            generated_text = generator[0]['generated_text']
            st.write(f"Here's the Generated Text:\n{generated_text}")


if __name__ == '__main__':
    main()
