import gradio as gr
import googleapiclient.discovery
from youtube_search import YoutubeSearch
import re
import json
import google.generativeai as genai
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Set up YouTube API key
API_KEY = "<YOUTUBE_API>"
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=API_KEY)

# Set up Gemini API key
genai.configure(api_key="<GEMINI_API>")

def extract_video_id(url):
    """Extract video ID from YouTube URL."""
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return match.group(1) if match else None

def get_video_title(video_id):
    """Fetch video title from YouTube API."""
    request = youtube.videos().list(part="snippet", id=video_id)
    response = request.execute()
    return response["items"][0]["snippet"]["title"] if response["items"] else None

def find_similar_videos(title):
    """Search YouTube for videos with a similar title."""
    results = YoutubeSearch(title, max_results=5).to_json()
    return json.loads(results)["videos"]

def get_video_comments(video_id):
    """Fetch top comments from a video."""
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=10
        )
        response = request.execute()
        for item in response.get("items", []):
            comments.append(item["snippet"]["topLevelComment"]["snippet"]["textDisplay"])
    except Exception as e:
        print(f"Error fetching comments: {e}")
    return comments

def analyze_sentiment(comments):
    """Perform basic sentiment analysis on comments."""
    positive, negative, neutral = 0, 0, 0
    for comment in comments:
        if any(word in comment.lower() for word in ["good", "great", "love", "awesome", "fantastic"]):
            positive += 1
        elif any(word in comment.lower() for word in ["bad", "terrible", "hate", "awful", "worst"]):
            negative += 1
        else:
            neutral += 1
    return positive, negative, neutral

def generate_sentiment_graph(positive, negative, neutral):
    """Generate a sentiment analysis graph as a base64 image."""
    labels = ["Positive", "Negative", "Neutral"]
    values = [positive, negative, neutral]
    colors = ['green', 'red', 'gray']
    
    fig, ax = plt.subplots()
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Count")
    ax.set_title("Sentiment Analysis of Comments")
    
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    buf.close()
    return f"<img src='data:image/png;base64,{img_base64}' style='width:100%;'/>"

def summarize_comments(comments):
    """Summarize a list of comments using Gemini AI."""
    if not comments:
        return "No comments available."
    text = " ".join(comments)
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(f"Summarize the following comments: {text}")
    return response.text if response else "Failed to generate summary."

def process_youtube_video(url):
    """Main function to process YouTube video and related comments."""
    video_id = extract_video_id(url)
    if not video_id:
        return "Invalid URL. Please enter a valid YouTube video URL."
    
    title = get_video_title(video_id)
    if not title:
        return "Could not fetch video title."
    
    similar_videos = find_similar_videos(title)
    result_data = []
    
    for video in similar_videos:
        vid_id = video["id"]
        vid_title = video["title"]
        channel = video["channel"]
        comments = get_video_comments(vid_id)
        summary = summarize_comments(comments)
        positive, negative, neutral = analyze_sentiment(comments)
        sentiment_graph = generate_sentiment_graph(positive, negative, neutral)
        
        result_data.append({
            "Title": vid_title,
            "Creator": channel,
            "Comments Summary": summary,
            "Sentiment Graph": sentiment_graph
        })
    
    return result_data

def display_results(url):
    results = process_youtube_video(url)
    if isinstance(results, str):
        return results
    
    formatted_results = """<div style='font-family: Arial, sans-serif;'>"""
    for res in results:
        formatted_results += f"""
        <div style='border: 2px solid #ddd; padding: 10px; margin: 10px; border-radius: 10px; background-color: #333; color: white;'>
            <h3>{res['Title']}</h3>
            <p><strong>Creator:</strong> {res['Creator']}</p>
            <p><strong>Comments Summary:</strong> {res['Comments Summary']}</p>
            <div>{res['Sentiment Graph']}</div>
        </div>
        """
    formatted_results += "</div>"
    
    return formatted_results

# Gradio Interface
demo = gr.Interface(
    fn=display_results,
    inputs=gr.Textbox(label="YouTube Video URL"),
    outputs=gr.HTML(),
    title="YouTube Video Comment Summarizer",
    description="Enter a YouTube video URL to find similar videos, summarize their comments, and analyze sentiment."
)

demo.launch()
