import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine, Sawtooth

# Function to generate a random bass tone (low frequency sine wave)
def generate_bass(filename, duration_ms=2000, frequency=60):
    bass_tone = Sine(frequency).to_audio_segment(duration=duration_ms)  # Generate a sine wave for bass
    bass_tone.export(filename, format="mp3")
    print(f"Bass sound saved as {filename}")

# Function to generate a random synth sound (sawtooth wave)
def generate_synth(filename, duration_ms=2000, frequency=440):
    synth_tone = Sawtooth(frequency).to_audio_segment(duration=duration_ms)  # Generate a sawtooth wave for synth
    synth_tone.export(filename, format="mp3")
    print(f"Synth sound saved as {filename}")

# Example usage: Generate 5 random bass and synth sounds
import os
import random

output_dir_bass = 'data/bass/'
output_dir_synth = 'data/synth/'
os.makedirs(output_dir_bass, exist_ok=True)
os.makedirs(output_dir_synth, exist_ok=True)

for i in range(5):
    # Randomize duration and frequency for variation
    bass_duration = random.randint(1000, 3000)  # Random duration between 1-3 seconds
    synth_duration = random.randint(1000, 3000)
    bass_frequency = random.randint(40, 120)  # Low frequencies for bass sounds
    synth_frequency = random.randint(300, 600)  # Mid-range frequencies for synth sounds
    
    # Generate and save bass and synth sounds
    generate_bass(f"{output_dir_bass}bass_{i+1}.mp3", duration_ms=bass_duration, frequency=bass_frequency)
    generate_synth(f"{output_dir_synth}synth_{i+1}.mp3", duration_ms=synth_duration, frequency=synth_frequency)
