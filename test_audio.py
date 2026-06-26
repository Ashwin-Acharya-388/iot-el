import pygame
import time
import os

def test_audio(file_path):
    """
    Plays an audio file to test the audio output.
    Ensure your Pagaria KN330 is connected and set as the default audio output device.
    """
    print("--- Audio Testing Script ---")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: Could not find audio file at '{file_path}'")
        print("Please provide a valid path to a .wav or .mp3 file.")
        return

    # Initialize pygame mixer
    try:
        pygame.mixer.init()
        print("Pygame mixer initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize audio mixer: {e}")
        return

    # Load and play the audio file
    try:
        print(f"Loading '{file_path}'...")
        pygame.mixer.music.load(file_path)
        
        print("Playing audio... (Press Ctrl+C to stop)")
        pygame.mixer.music.play()
        
        # Keep the script running while the audio plays
        while pygame.mixer.music.get_busy():
            time.sleep(1)
            
        print("Playback finished.")
    except Exception as e:
        print(f"Error during playback: {e}")
    finally:
        pygame.mixer.quit()

if __name__ == "__main__":
    # You can change this to the path of any .wav or .mp3 file on your laptop
    # For testing, make sure you have a sample audio file in the same directory.
    sample_file = "sample.wav" 
    
    print(f"To test, ensure you have a valid audio file named '{sample_file}' in the current directory.")
    test_audio(sample_file)
