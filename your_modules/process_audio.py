import whisper
from pyannote.audio import Pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load Whisper model
whisper_model = whisper.load_model("base")

# Load the pre-trained emotion classifier from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")

# Initialize pyannote's pretrained diarization model
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="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")

# Function for transcription using Whisper
def transcribe_audio(file_path):
    result = whisper_model.transcribe(file_path)
    return result['text']

# Function for speaker diarization
def diarize_audio(file_path):
    diarization = pipeline({'uri': 'filename', 'audio': file_path})
    # Convert diarization results to a more readable format
    diarization_result = []
    for speech_turn, _, speaker in diarization.itertracks(yield_label=True):
        diarization_result.append({'start': speech_turn.start, 'end': speech_turn.end, 'speaker': speaker})
    return diarization_result

# Function to predict emotion from text
def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    return model.config.id2label[predicted_class]

# Main function to process audio
def process_audio(file_path):
    # Transcription
    transcription = transcribe_audio(file_path)
    
    # Speaker diarization
    diarization = diarize_audio(file_path)
    
    # Emotion classification on transcription
    emotion = predict_emotion(transcription)
    
    # Combine results
    return {
        'transcription': transcription,
        'diarization': diarization,
        'emotion': emotion
    }
