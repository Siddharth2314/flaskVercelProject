from flask import Flask, render_template, request
import os
import shutil

import librosa
import openai
import soundfile as sf
# import youtube_dl
# from youtube_dl.utils import DownloadError
from pytube import YouTube
from pydub import AudioSegment


API_KEY = "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"
outputs_dir = "outputs/"
openai.api_key = API_KEY

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        videoUrl = request.form['VideoURL']
        long_summary, short_summary = summarize_youtube_video(videoUrl, outputs_dir)
        message = {'youtube_url': videoUrl, 'long_summary': long_summary, 'short_summary': short_summary}
        print(videoUrl)
        if videoUrl == '':
            return render_template('index.html', message="Please enter required field.")
        return render_template('success.html', message)

# Utility function

def find_audio_files(path, extension=".mp3"):
    """Recursively find all files with extension in path."""
    audio_files = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(extension):
                audio_files.append(os.path.join(root, f))
    return audio_files

# Download youtube audio

def youtube_to_mp3(youtube_url: str, output_dir: str) -> str:
    """Download the audio from a YouTube video, save it to output_dir as an .mp3 file.

    Returns the filename of the saved video.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Downloading video from {youtube_url}")

    try:
        # Download YouTube video
        yt = YouTube(youtube_url)
        video_stream = yt.streams.filter(only_audio=True).first()
        video_path = video_stream.download(output_dir, filename="temp_video")

        # Convert video to MP3 using pydub
        audio_path = os.path.join(output_dir, f"{yt.title}.mp3")

        audio = AudioSegment.from_file(video_path)
        audio.export(audio_path, format="mp3")

        # Clean up temporary video file
        os.remove(video_path)

        return audio_path

    except Exception as e:
        print(f"Error downloading or converting video from {youtube_url}: {e}")
        return None


# Chunk the audio
    # Chunking is necessary in the case where we have very long audio files, since both whisper and ChatGPT have limits of how much audio/text you can process in one go. It is not necessary for shorter videos.

def chunk_audio(filename, segment_length: int, output_dir):
    """Segment length is in seconds"""
    
    print(f"Chunking audio to {segment_length} second segments...")
    
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
        
    # Load audio file
    audio, sr = librosa.load(filename, sr=44100)
    
    # Calculate duration in seconds
    duration = librosa.get_duration(y=audio, sr=sr)
    
    # Calculate number of segments
    num_segments = int(duration / segment_length) + 1
    
    print(f"Chunking {num_segments} chunks...")
    
    # Iterate through segments and save them
    for i in range(num_segments):
        start = i * segment_length * sr
        end = (i + 1) * segment_length * sr
        segment = audio[start:end]
        sf.write(os.path.join(output_dir, f"segment_{i}.mp3"), segment, sr)
        
    chunk_audio_files = find_audio_files(output_dir)
    return sorted(chunk_audio_files)

# Speech2text
# Here we use OpenAI's whisper model to trancribe audio files to text.

def transcribe_audio(audio_files: list, output_file=None, model="whisper-1") -> list:
    print("Converting audio to text...")
    
    transcripts = []
    for audio_file in audio_files:
        with open(audio_file, "rb") as audio:
            response = openai.Audio.transcribe(model, audio)
            transcripts.append(response["text"])
    
    if output_file is not None:
        # Save all transcriptions to a .txt file
        with open(output_file, "w") as file:
            for transcript in transcripts:
                file.write(transcript + "\n")
    
    return transcripts

# Summarize
# Here we ask chatGPT to take the raw transcripts and transcribe for us to short bullet points.

def summarize(chunks: list[str], system_prompt: str, model="gpt-3.5-turbo", output_file=None):
    
    print(f"Summarizing with {model=}")
    
    summaries = []
    for chunk in chunks:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk},
            ],
        )
        summary = response["choices"][0]["message"]["content"]
        summaries.append(summary)
        
    if output_file is not None:
        #save all transcripts to a .txt file
        with open(output_file, "w") as file:
            for summary in summaries:
                file.write(summary + "\n")
    
    return summaries

# Putting it all together

def summarize_youtube_video(youtube_url, outputs_dir):
    raw_audio_dir = f"{outputs_dir}/raw_audio/"
    chunks_dir = f"{outputs_dir}/chunks"
    transcripts_file = f"{outputs_dir}/transcripts.txt"
    summary_file = f"{outputs_dir}/summary.txt"
    segment_length = 10 * 60  # chunk to 10 minute segments

    if os.path.exists(outputs_dir):
        # delete the outputs_dir folder and start from scratch
        shutil.rmtree(outputs_dir)
        os.mkdir(outputs_dir)

    # download the video using youtube-dl
    audio_filename = youtube_to_mp3(youtube_url, output_dir=raw_audio_dir)

    # chunk each audio file to shorter audio files (not necessary for shorter videos...)
    chunked_audio_files = chunk_audio(
        audio_filename, segment_length=segment_length, output_dir=chunks_dir
    )

    # transcribe each chunked audio file using whisper speech2text
    transcriptions = transcribe_audio(chunked_audio_files, transcripts_file)

    # summarize each transcription using chatGPT
    system_prompt = """
    You are a helpful assistant that summarizes youtube videos.
    You are provided chunks of raw audio that were transcribed from the video's audio.
    Summarize the current chunk to succint and clear bullet points of its contents.
    """
    summaries = summarize(
        transcriptions, system_prompt=system_prompt, output_file=summary_file
    )

    system_prompt_tldr = """
    You are a helpful assistant that summarizes youtube videos.
    Someone has already summarized the video to key points.
    Summarize the key points to one or two sentences that capture the essence of the video.
    """
    # put the entire summary to a single entry
    long_summary = "\n".join(summaries)
    short_summary = summarize(
        [long_summary], system_prompt=system_prompt_tldr, output_file=summary_file
    )[0]

    return long_summary, short_summary
