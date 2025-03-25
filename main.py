import os
import threading 
import cv2
from deepface import DeepFace
import time
from gtts import gTTS
import pygame
import unicodedata

# Suppress TensorFlow informational logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize video capture with the first camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Set frame dimensions
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # Height

# Initialize pygame for playing sound
pygame.mixer.init()

# Global variables
face_match = False
matched_person = ""
frame_to_check = None
reference_images = []
reference_names = []  # To store names without accents

def remove_accents(input_str):
    # Normalize string to remove accents
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

# Load all reference images from the 'images' directory
reference_dir = "images"
for filename in os.listdir(reference_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(reference_dir, filename)
        print(f"Attempting to load image: {img_path}")  # Debugging output
        
        # Try to read the image
        img = cv2.imread(img_path)
        if img is not None:  # Check if the image was loaded successfully
            person_name_with_accent = os.path.splitext(filename)[0]  # Original name
            person_name_without_accent = remove_accents(person_name_with_accent)  # Name without accent
            reference_images.append((img, person_name_with_accent))  # Add name with accent
            reference_names.append(person_name_without_accent)  # Add name without accent
            
            print(f"Loaded {person_name_with_accent} successfully.")  # Informative message
        else:
            print(f"Warning: Unable to load image {filename}. Check if the file exists and if it's readable.")

def check_face():
    global face_match, matched_person, frame_to_check
    
    while True:
        if frame_to_check is not None:
            face_match = False
            for reference_img, person_name in reference_images:
                try:
                    if DeepFace.verify(frame_to_check, reference_img.copy())['verified']:
                        face_match = True
                        matched_person = person_name
                        break
                except ValueError as e:
                    print(f"Error during verification: {e}")  # Debugging info
                    continue
            frame_to_check = None

def play_welcome_message(person_name):
    # Generate the speech in Portuguese with the correct accent
    tts = gTTS(f"Bem-vindo, {person_name}.", lang='pt')
    audio_file = "welcome.mp3"
    tts.save(audio_file)
    
    # Play the audio using pygame
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    
    while pygame.mixer.music.get_busy():  # Wait until the audio is finished
        pygame.time.Clock().tick(10)
    
    # Remove the audio file after playback is done
    pygame.mixer.music.unload()  # Unload the music before removing the file
    os.remove(audio_file)

# Start the face checking thread
threading.Thread(target=check_face, daemon=True).start()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Pass the frame to the face checking thread
    if frame_to_check is None:
        frame_to_check = frame.copy()
    
    # Display the resulting frame
    if face_match:
        cv2.putText(frame, f"Match face {matched_person}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Live Face Recognition', frame)
        cv2.waitKey(1)  # Refresh the display
        play_welcome_message(matched_person)  # Play the welcome message
        time.sleep(1)  # Pause for 1 second
        face_match = False  # Reset the flag after playing the message
    else:
        cv2.putText(frame, "No Match", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Live Face Recognition', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and clean up
cap.release()
cv2.destroyAllWindows()
