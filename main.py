from PIL import Image
import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine
from tqdm import tqdm
from scipy.fft import fft
import wave
import contextlib

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

def audio_to_image(audio_path, output_image_path):
    # Load the audio
    with contextlib.closing(wave.open(audio_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        audio_data = np.frombuffer(f.readframes(frames), dtype=np.int16)

    # Parameters
    min_freq = 200   # Minimum frequency (Hz)
    max_freq = 2000  # Maximum frequency (Hz)
    duration_ms = 50 # Duration of each tone (ms)

    # Calculate the number of samples per segment
    segment_length = int(rate * (duration_ms / 1000.0))

    # Calculate the number of segments
    num_segments = int(duration * 1000 / duration_ms)

    # Initialize an empty list to store pixel values
    pixel_values = []

    # Initialize the progress bar
    with tqdm(total=num_segments, desc="Converting audio to image") as pbar:
        for i in range(num_segments):
            # Extract the segment of audio data
            segment = audio_data[i * segment_length:(i + 1) * segment_length]
            # Perform FFT to find the dominant frequency
            freqs = np.fft.fftfreq(len(segment))
            fft_values = np.abs(fft(segment))
            dominant_freq = abs(freqs[np.argmax(fft_values)])

            # Map the frequency to a pixel value
            pixel_value = 255 * (dominant_freq - min_freq) / (max_freq - min_freq)
            pixel_value = int(np.clip(pixel_value, 0, 255))

            # Append the pixel value to the list
            pixel_values.append(pixel_value)

            # Update the progress bar
            pbar.update(1)

    # Convert the list of pixel values to a 2D array
    pixels = np.array(pixel_values).reshape((100, 100))

    # Create an image from the 2D array of pixels
    img = Image.fromarray(pixels.astype(np.uint8))
    img.save(output_image_path)

if __name__ == "__main__":
    import sys

    print("What would you like to do?")
    print("1: Convert image to audio")
    print("2: Convert audio to image")

    choice = input("Enter your choice (1 or 2): ")

    if choice == "1":
        image_path = input("Enter the path to the input image: ")
        output_audio_path = input("Enter the path to save the output audio: ")
        image_to_audio(image_path, output_audio_path)
    elif choice == "2":
        audio_path = input("Enter the path to the input audio: ")
        output_image_path = input("Enter the path to save the output image: ")
        audio_to_image(audio_path, output_image_path)
    else:
        print("Invalid choice. Please enter 1 or 2.")
