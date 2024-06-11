from PIL import Image
import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine
from tqdm import tqdm

def image_to_audio(image_path, output_audio_path):
    # Load the image
    img = Image.open(image_path).convert('L')  # Convert image to grayscale
    img = img.resize((100, 100))  # Resize for simplicity
    pixels = np.array(img)

    # Create an empty audio segment
    audio = AudioSegment.silent(duration=0)

    # Map pixel brightness to frequency (pitch)
    max_freq = 2000  # Maximum frequency (Hz)
    min_freq = 200   # Minimum frequency (Hz)
    duration_ms = 50 # Duration of each tone (ms)

    # Initialize the progress bar
    total_pixels = pixels.shape[0] * pixels.shape[1]
    with tqdm(total=total_pixels, desc="Converting image to audio") as pbar:
        for row in pixels:
            for pixel in row:
                # Normalize pixel value to range between min_freq and max_freq
                frequency = min_freq + (max_freq - min_freq) * (pixel / 255.0)
                # Generate a sine wave for the corresponding frequency
                sine_wave = Sine(frequency).to_audio_segment(duration=duration_ms)
                # Append the sine wave to the audio segment
                audio += sine_wave
                # Update the progress bar
                pbar.update(1)

    # Export the generated audio
    audio.export(output_audio_path, format="wav")

# Example usage
image_to_audio('input_image.jpg', 'output_audio.wav')
