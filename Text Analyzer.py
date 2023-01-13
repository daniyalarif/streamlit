import streamlit as st

# NLP Packages 
import spacy
spacy.cli.download("en_core_web_sm")
from textblob import TextBlob
from gensim.summarization import summarize

# Sumy Pkgs
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Summary Function
def sumy_summarizer(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document,3)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result    

def text_analyzer(my_text):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(my_text)
    
    tokens = [token.text for token in docx]
    allData = ['"Tokens":{},\n"Lemma":{}'.format(token.text, token.lemma_) for token in docx]
    return allData

@st.cache
def entity_analyzer(my_text):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(my_text)
    tokens = [token.text for token in docx]
    entities = [(entity.text, entity.label_) for entity in docx.ents]
    allData = ['"Tokens":{},\n"Entities":{}'.format(tokens,entities)]
    return allData


# Pkgs

def main():
    """NLP App with Streamlit"""
    st.title("NLPiffy with Streamlit")
    st.subheader("Natural language processing on the Go")
    
    # Tokenization
    if st.checkbox("Show tokens and Lemma"):
        st.subheader("Tokenize your text")
        message = st.text_area("Enter your text","Type here")
        if st.button("Analyze"):
            nlp_result = text_analyzer(message)
            st.json(nlp_result)
        
        
    # Named Entity
    if st.checkbox("Show Named Entities"):
        st.subheader("Extract Entities From Your Text")
        message = st.text_area("Enter your text","Type here")
        if st.button("Extract"):
            nlp_result = entity_analyzer(message)
            st.json(nlp_result)
    
    #Sentiment Analysis
    if st.checkbox("Show Sentiment Analysis"):
        st.subheader("Sentiment of Your Text")
        message = st.text_area("Enter your text","Type here")
        if st.button("Analyze"):
            blob = TextBlob(message)
            result_sentiment = blob.sentiment
            st.success(result_sentiment)
       
    # Text Summarization
    if st.checkbox("Show Text Summarization"):
        st.subheader("Summarize of Your Text")
        message = st.text_area("Enter your text","Type here")
        summary_options = st.selectbox("Choice your summarizer",("gensim","sumy"))
        if st.button("Summarize"):
            if summary_options == 'gensim':
                st.text("Using Gensim...")
                summary_result = summarize(message)
            elif summary_options == 'sumy':
                st.text("Using Sumy...")
                summary_result = sumy_summarizer(message)
            else:
                st.warning("Using default summarizer")
                st.text("Using Gensim")
                summary_result = summarize(message)
                
            st.success(summary_result)
            
    st.sidebar.subheader("About the APP")
    st.sidebar.text("NLP App with Streamlit")
    st.sidebar.info("Kudos to the streamlit team")
    
            
if __name__ == '__main__':
    main()