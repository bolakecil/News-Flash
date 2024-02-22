from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup
from transformers import BartTokenizer, BartForConditionalGeneration

app = Flask(__name__)

# Load pre-trained DistilBART model and tokenizer
model_name = "sshleifer/distilbart-cnn-12-6"  # DistilBART model
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

def extract_news_details(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extracting the title
    title = soup.title.string

    # Extracting main content
    paragraphs = soup.find_all('p')
    content = ' '.join([p.text for p in paragraphs])

    return {
        'title': title,
        'content': content
    }

def summarize_text(text, max_len=200):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_len, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

def summarize_news_from_url(url):
    details = extract_news_details(url)
    content_summary = summarize_text(details['content'])

    return {
        'title': details['title'],  # Directly using the extracted title
        'summary': content_summary
    }

@app.route("/", methods=['GET', 'POST'])
def index():
    title = None
    summary = None
    if request.method == 'POST':
        url = request.form['url']
        news_summary = summarize_news_from_url(url)
        title = news_summary['title']
        summary = news_summary['summary']
    return render_template("index.html", title=title, summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
