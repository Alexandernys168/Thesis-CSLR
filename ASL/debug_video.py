import cv2
import os

VIDEO_PATH = r"a:\Thesis-CSLR\ASL\1\wlasl-complete\videos\70212.mp4"

def inspect():
    if not os.path.exists(VIDEO_PATH):
        print("File Missing")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Failed to open video")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video: {VIDEO_PATH}")
    print(f"Total Frames: {total_frames}")
    print(f"Dimensions: {width}x{height}")
    print(f"FPS: {fps}")
    
    # Try reading the first frame
    ret, frame = cap.read()
    print(f"Read Match 1: {ret}")
    
    cap.release()

if __name__ == "__main__":
    inspect()
