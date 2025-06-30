from speech2text.whisper import WhisperTranscriber

transcriber = WhisperTranscriber()
results = transcriber.transcribe_directory("datasets/mp3", "output")