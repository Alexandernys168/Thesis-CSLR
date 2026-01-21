import mediapipe as mp
import os
print("MediaPipe imported")
try:
    print(f"File: {mp.__file__}")
    print(f"Path: {mp.__path__}")
    print(f"Dir: {dir(mp)}")
    import mediapipe.python.solutions as solutions
    print("Direct import of solutions worked")
    print("Solutions:", solutions)
except Exception as e:
    print("Error:", e)
