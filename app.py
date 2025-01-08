import gradio as gr
import whisper
import torch
import json
import spaces
from datetime import timedelta
import os
import zipfile
from pathlib import Path

def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format"""
    td = timedelta(seconds=seconds)
    hours = td.seconds//3600
    minutes = (td.seconds//60)%60
    seconds = td.seconds%60
    milliseconds = td.microseconds//1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def save_files(text, srt, json_data, base_name):
    """Save transcription in different formats and create zip"""
    # Create output directory if it doesn't exist
    output_dir = Path("transcriptions")
    output_dir.mkdir(exist_ok=True)
    
    # Generate filenames
    base_name = Path(base_name).stem
    txt_path = output_dir / f"{base_name}.txt"
    srt_path = output_dir / f"{base_name}.srt"
    json_path = output_dir / f"{base_name}.json"
    zip_path = output_dir / f"{base_name}_all.zip"
    
    # Save individual files
    txt_path.write_text(text)
    srt_path.write_text(srt)
    json_path.write_text(json_data)
    
    # Create ZIP file
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(txt_path, txt_path.name)
        zipf.write(srt_path, srt_path.name)
        zipf.write(json_path, json_path.name)
    
    return str(txt_path), str(srt_path), str(json_path), str(zip_path)

@spaces.GPU
def transcribe(audio_file):
    # Load the Whisper model
    model = whisper.load_model("large-v3-turbo")
    
    # Transcribe the audio
    result = model.transcribe(audio_file)
    
    # Format as plain text
    text_output = result["text"]
    
    # Format as JSON
    json_output = json.dumps(result, indent=2)
    
    # Format as SRT
    srt_output = ""
    for i, segment in enumerate(result["segments"], 1):
        start_time = format_timestamp(segment["start"])
        end_time = format_timestamp(segment["end"])
        text = segment["text"].strip()
        srt_output += f"{i}\n{start_time} --> {end_time}\n{text}\n\n"
    
    # Save files and get paths
    txt_file, srt_file, json_file, zip_file = save_files(
        text_output, srt_output, json_output, 
        os.path.basename(audio_file)
    )
    
    return (
        txt_file, srt_file, json_file, zip_file, text_output, srt_output, json_output
    )

# Create the Gradio interface
demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath", label="Upload Audio"),
    outputs=[
        gr.File(label="Download TXT"),
        gr.File(label="Download SRT"),
        gr.File(label="Download JSON"),
        gr.File(label="Download All (ZIP)"),
        gr.Textbox(label="Transcription", lines=5),
        gr.Textbox(label="SRT Format"),
        gr.JSON(label="JSON Output")
    ],
    title="Audio Transcription with Whisper",
    description="Upload an audio file to transcribe it into text, SRT, and JSON formats using OpenAI's Whisper model. You can download the results in different formats or get everything in a ZIP file."
)

if __name__ == "__main__":
    demo.launch(share=True)