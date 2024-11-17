from flask import Flask, render_template, request, jsonify
import os
from your_modules.process_audio import process_audio
from your_modules.sentiment_analysis import analyze_sentiment

app = Flask(__name__)

# Folder for uploaded audio files
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Save the uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    # Process the audio (transcription, speaker diarization, sentiment analysis, emotion detection)
    result = process_audio(filepath)

    # Get sentiment analysis
    sentiment = analyze_sentiment(result['transcription'])
    
    # Combine results
    result['sentiment'] = sentiment['sentiment']
    result['emotion'] = result['emotion']

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
