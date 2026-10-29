import cv2
from pathlib import Path
import numpy as np
import os

def label_face_filename(face, filename):
    text = os.path.splitext(os.path.basename(filename))[0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (5, face.shape[0] - 10)
    fontScale = 0.5
    face = face.copy() # numpy array issue
    # Black outline for readability on bright faces
    cv2.putText(face, text, org, font, fontScale, (0, 0, 0), 3, cv2.LINE_AA)
    # White text on top
    cv2.putText(face, text, org, font, fontScale, (255, 255, 255), 1, cv2.LINE_AA)

    return face
