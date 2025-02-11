import gradio as gr
import googleapiclient.discovery
import re
import torch
from transformers import pipeline
import matplotlib.pyplot as plt
from collections import Counter
import google.generativeai as genai
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# YouTube API Key (Replace with your own)
API_KEY = "<YOUTUBE_API_KEY>"
GEMINI_API_KEY = "<GEMINI_API_KEY>"

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-pro-latest")

# Sentiment Analysis Pipeline
sentiment_pipeline = pipeline("sentiment-analysis", 
                             model="nlptown/bert-base-multilingual-uncased-sentiment",
                             device=0 if torch.cuda.is_available() else -1)

# Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_video_id(url):
    match = re.search(r"v=([\w-]+)", url)
    return match.group(1) if match else None

def fetch_comments(video_url):
    video_id = extract_video_id(video_url)
    if not video_id:
        return "Invalid YouTube URL"

    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=API_KEY)
    comments = []
    next_page_token = None
    
    while len(comments) < 100:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, 100 - len(comments)),
            pageToken=next_page_token
        )
        response = request.execute()
        
        for item in response.get("items", []):
            comments.append(item["snippet"]["topLevelComment"]["snippet"]["textOriginal"])
            if len(comments) >= 100:
                break
        
        if len(comments) >= 100 or not response.get("nextPageToken"):
            break
        next_page_token = response.get("nextPageToken")
    
    return comments[:100]


def summarize_comments(comments):
    try:
        joined_comments = " ".join(comments[:20])
        response = gemini_model.generate_content(
            f"Summarize these YouTube comments in 3 sentences. Focus on key themes and sentiment:\n\n{joined_comments}"
        )
        return response.text
    except Exception as e:
        return f"Summarization failed: {str(e)}"

def classify_sentiment(comments):
    sentiment_map = {
        1: "Extremely Negative",
        2: "Negative",
        3: "Neutral",
        4: "Positive",
        5: "Extremely Positive"
    }

    results = sentiment_pipeline(comments)
    return [sentiment_map[int(res['label'][0])] for res in results]

def plot_histogram(sentiments):
    counts = Counter(sentiments)
    labels, values = zip(*counts.items())

    plt.figure(figsize=(8,5))
    plt.bar(labels, values, color=['#ff4444', '#ffbb33', '#aaaaaa', '#99cc00', '#00c851'])
    plt.xlabel("Sentiment Category")
    plt.ylabel("Number of Comments")
    plt.title("Sentiment Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig("histogram.png")
    return "histogram.png"

def train_rag(comments):
    embeddings = embedding_model.encode(comments, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings, comments

def answer_question(question, index, embeddings, comments):
    try:
        question_embedding = embedding_model.encode([question], convert_to_numpy=True)
        distances, indices = index.search(question_embedding, k=3)
        context = "\n".join([comments[i] for i in indices[0]])
        
        response = gemini_model.generate_content(
            f"Answer this question in 3 sentences based on the given context:\n\n"
            f"Question: {question}\n\n"
            f"Comments:\n{context}"
        )
        return response.text
    except Exception as e:
        return f"Question answering failed: {str(e)}"

# Global RAG storage
rag_index = None
rag_embeddings = None
rag_comments = None

def analyze_youtube_comments(video_url):
    global rag_index, rag_embeddings, rag_comments
    
    comments = fetch_comments(video_url)
    if isinstance(comments, str) or len(comments) == 0:
        return "No comments found", None, None, None
        
    sentiments = classify_sentiment(comments)
    histogram = plot_histogram(sentiments)
    summary = summarize_comments(comments)
    rag_index, rag_embeddings, rag_comments = train_rag(comments)
    
    return (
        f"Analyzed {len(comments)} comments\n"
        f"Dominant sentiment: {Counter(sentiments).most_common(1)[0][0]}",
        histogram,
        summary,
        "Similar videos feature temporarily disabled"
    )

def query_rag(question):
    if not rag_index or not question.strip():
        return "Please analyze a video first and enter a valid question"
    return answer_question(question, rag_index, rag_embeddings, rag_comments)

# Gradio Interface
with gr.Blocks(title="YouTube Comment Analyst", theme=gr.themes.Soft()) as iface:
    gr.Markdown("# YouTube Comment Analysis with Gemini")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Textbox(label="YouTube Video URL", placeholder="Enter video URL...")
            analyze_btn = gr.Button("Analyze Comments", variant="primary")
            
            gr.Markdown("## Analysis Results")
            sentiment_output = gr.Textbox(label="Sentiment Summary", interactive=False)
            histogram_output = gr.Image(label="Sentiment Distribution", interactive=False)
            summary_output = gr.Textbox(label="Comment Summary", interactive=False)
            
        with gr.Column():
            gr.Markdown("## Ask About Comments")
            question_input = gr.Textbox(label="Your Question", placeholder="What are people saying about...?")
            query_btn = gr.Button("Get Answers", variant="secondary")
            answer_output = gr.Textbox(label="Gemini's Response", interactive=False)
            
    analyze_btn.click(
        analyze_youtube_comments,
        inputs=video_input,
        outputs=[sentiment_output, histogram_output, summary_output]
    )
    
    query_btn.click(
        query_rag,
        inputs=question_input,
        outputs=answer_output
    )

if __name__ == "__main__":
    iface.launch(server_port=7860, share=True)
