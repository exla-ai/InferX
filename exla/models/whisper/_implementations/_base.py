class Whisper_Base:
    def __init__(self):
        super().__init__()

    def transcribe(self, audio_path, language=None, task=None):
        """
        Transcribe audio file to text.
        
        Args:
            audio_path (str): Path to the audio file or a numpy array of audio samples.
            language (str, optional): Language of the audio. If None, language is auto-detected.
            task (str, optional): Task to perform. Either 'transcribe' or 'translate'. Defaults to 'transcribe'.
            
        Returns:
            dict: Transcription result containing text and other metadata.
        """
        print(f"Running transcription on {self.__class__.__name__}")
        return {"text": "test_transcription"} 