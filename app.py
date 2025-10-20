# ----------------------------------------------------------------------------
#                                LIBRARIES
# ----------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import pdfplumber
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings

# Suppress unwanted warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", "Cannot set gray stroke color", UserWarning)

# ----------------------------------------------------------------------------
#                               PAGE CONFIGURATION
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="BPCL Annual Report Analyzer",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------------
#                               CUSTOM CSS (BPCL THEME)
# ----------------------------------------------------------------------------
def local_css():
    st.markdown("""
        <style>
        body { background-color: #f5faff; color: #1b2a49; font-family: 'Inter', sans-serif; }
        .main .block-container { padding: 2.0rem 2.5rem; }
        [data-testid="stSidebar"] { background-color: #007acc; color: white; }
        [data-testid="stSidebar"] * { color: white !important; }
        h1,h2,h3,h4,h5,h6 { color:#004b87; font-weight:600; }
        .card { background-color:#ffffff; border-radius:10px; border:1px solid #dce3f0; padding:1.6rem; margin-bottom:1.2rem; box-shadow:0 4px 8px rgba(0,0,0,0.04); }
        .stMetric label { color:#004b87 !important; }
        .stMetric div[data-testid="metric-value"] { color:#007acc !important; font-size:1.8rem; font-weight:bold; }
        .stTextArea textarea { background-color:#f2f7fb; color:#1b2a49; border-radius:8px; border:1px solid #ccd9ea; }
        .block-container .stPlotlyChart, .block-container .stPyplot { background: transparent; }
        </style>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
#                               INITIAL SETUP
# ----------------------------------------------------------------------------
@st.cache_resource
def download_nltk_data():
    for resource in ['punkt', 'stopwords']:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource=='punkt' else f'corpora/{resource}')
        except Exception:
            nltk.download(resource, quiet=True)

# ----------------------------------------------------------------------------
#                               HELPER FUNCTIONS
# ----------------------------------------------------------------------------
@st.cache_data
def load_and_extract_text(file):
    """Read uploaded PDF and return text + page count."""
    all_text = ""
    with pdfplumber.open(file) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"
    return all_text, page_count

@st.cache_data
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w)>1]
    return ' '.join(tokens)

@st.cache_data
def analyze_sentiment(raw_text):
    sentences = sent_tokenize(raw_text)
    sentiments = []
    for s in sentences:
        blob = TextBlob(s)
        sentiments.append({
            "Sentence": s,
            "Polarity": blob.sentiment.polarity,
            "Subjectivity": blob.sentiment.subjectivity
        })
    df = pd.DataFrame(sentiments)
    df['Sentiment'] = df['Polarity'].apply(lambda x: 'Positive' if x>0.05 else ('Negative' if x<-0.05 else 'Neutral'))
    return df

@st.cache_data
def get_topic_model(clean_text, num_topics):
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    doc_term_matrix = tfidf_vectorizer.fit_transform([clean_text])
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_model.fit(doc_term_matrix)
    return lda_model, tfidf_vectorizer, doc_term_matrix.shape

# ----------------------------------------------------------------------------
#                               PLOT STYLING
# ----------------------------------------------------------------------------
def style_plot(fig, ax, title):
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.set_title(title, color='#004b87', fontsize=16, pad=12)
    ax.xaxis.label.set_color('#004b87')
    ax.yaxis.label.set_color('#004b87')
    ax.tick_params(colors='#004b87')
    for spine in ax.spines.values():
        spine.set_visible(False)  # safer in Streamlit

# ----------------------------------------------------------------------------
#                               MAIN APP
# ----------------------------------------------------------------------------
def main():
    st.title("💠 BPCL Annual Report Analyzer")
    local_css()
    download_nltk_data()

    st.sidebar.title("BPCL Analyzer")
    st.sidebar.markdown("Analyze the 2024–25 Annual Report.")
    page_options = ["Overview", "Sentiment Analysis", "Word Analysis", "Topic Modeling"]
    selected_page = st.sidebar.radio("Navigate", page_options)
    st.sidebar.markdown("---")
    st.sidebar.markdown("👨‍💻 Developed by **Snehil**")

    uploaded_file = st.file_uploader("Upload BPCL Annual Report PDF", type="pdf")
    if uploaded_file:
        with st.spinner("Processing PDF..."):
            raw_text, page_count = load_and_extract_text(uploaded_file)
        clean_text = preprocess_text(raw_text)
        all_tokens = clean_text.split()
        df_sentiments = analyze_sentiment(raw_text)

        # -------------------- OVERVIEW --------------------
        if selected_page=="Overview":
            st.subheader("Document Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Pages", page_count)
            col2.metric("Characters", f"{len(raw_text):,}")
            col3.metric("Sentences", f"{len(df_sentiments):,}")
            col4.metric("Tokens", f"{len(all_tokens):,}")
            st.subheader("Preview Text (first 2500 chars)")
            st.text_area("", raw_text[:2500], height=350)

        # -------------------- SENTIMENT --------------------
        elif selected_page=="Sentiment Analysis":
            st.subheader("Sentiment Overview")
            avg_polarity = df_sentiments['Polarity'].mean()
            sentiment_counts = df_sentiments['Sentiment'].value_counts().reindex(['Positive','Neutral','Negative']).fillna(0)
            col1, col2 = st.columns([1,2])
            with col1:
                st.metric("Average Polarity", f"{avg_polarity:.3f}")
                fig_pie, ax_pie = plt.subplots(figsize=(6,6))
                colors = ['#007acc','#f4c542','#e63946']
                ax_pie.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
                ax_pie.axis('equal')
                style_plot(fig_pie, ax_pie, "Sentiment Composition")
                st.pyplot(fig_pie, clear_figure=True)
            with col2:
                st.subheader("Polarity & Subjectivity Distribution")
                fig_hist, ax_hist = plt.subplots(figsize=(10,5))
                sns.histplot(df_sentiments['Polarity'], bins=50, kde=True, ax=ax_hist, color="#007acc", label="Polarity")
                sns.histplot(df_sentiments['Subjectivity'], bins=50, kde=True, ax=ax_hist, color="#f4c542", label="Subjectivity")
                ax_hist.legend()
                style_plot(fig_hist, ax_hist, "Distribution of Sentiment Scores")
                st.pyplot(fig_hist, clear_figure=True)
            st.subheader("Example Sentences")
            st.write("**Most Positive:**")
            st.dataframe(df_sentiments.nlargest(5,'Polarity')[['Sentence','Polarity']])
            st.write("**Most Negative:**")
            st.dataframe(df_sentiments.nsmallest(5,'Polarity')[['Sentence','Polarity']])

        # -------------------- WORD ANALYSIS --------------------
        elif selected_page=="Word Analysis":
            st.subheader("Top 20 Frequent Words")
            freq_dist = nltk.FreqDist(all_tokens)
            df_freq = pd.DataFrame(freq_dist.most_common(20), columns=['Word','Count'])
            fig, ax = plt.subplots(figsize=(8,8))
            sns.barplot(x='Count', y='Word', data=df_freq, palette='Blues_r', ax=ax)
            style_plot(fig, ax, "Most Frequent Words")
            st.pyplot(fig, clear_figure=True)
            st.subheader("Word Cloud")
            wordcloud = WordCloud(width=900,height=500,background_color="white",colormap='Blues').generate(clean_text)
            fig_wc, ax_wc = plt.subplots(figsize=(10,6))
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis("off")
            st.pyplot(fig_wc, clear_figure=True)

        # -------------------- TOPIC MODELING --------------------
        elif selected_page=="Topic Modeling":
            st.subheader("Latent Topic Modeling")
            num_topics = st.slider("Number of Topics:", min_value=3, max_value=15, value=8)
            lda_model, tfidf_vectorizer, matrix_shape = get_topic_model(clean_text, num_topics)
            feature_names = tfidf_vectorizer.get_feature_names_out()
            st.write(f"TF-IDF Matrix Shape: {matrix_shape}")
            cols = st.columns(2)
            for i in range(num_topics):
                with cols[i%2]:
                    topic = lda_model.components_[i]
                    top_words_idx = topic.argsort()[:-8:-1]
                    top_words = [feature_names[j] for j in top_words_idx]
                    top_weights = topic[top_words_idx]
                    fig, ax = plt.subplots(figsize=(6,4))
                    sns.barplot(x=top_weights, y=top_words, ax=ax, palette="YlGnBu")
                    style_plot(fig, ax, f'Topic #{i+1}')
                    st.pyplot(fig, clear_figure=True)
    else:
        st.info("Upload a BPCL Annual Report PDF to start analysis.")

# ---------------------- RUN APP -------------------------
if __name__=="__main__":
    main()
