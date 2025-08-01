import tkinter as tk
from PIL import Image, ImageTk
import cv2

# Setup root window
root = tk.Tk()
root.title("Sign Language To Text Conversion")
root.geometry("1300x700")
root.configure(bg="white")

# Title
title = tk.Label(root, text="Intelligent Hand Sign Recognition and Linguistic Translation System",
                 font=("Helvetica", 20, "bold"), bg="white")
title.pack(pady=10)

# Frames: webcam + skeleton
frame_main = tk.Frame(root, bg="white")
frame_main.pack()

# Webcam frame (left)
webcam_label = tk.Label(frame_main, bg="black", width=600, height=400)
webcam_label.grid(row=0, column=0, padx=30)

# Skeleton canvas (right)
skeleton_canvas = tk.Canvas(frame_main, bg="white", width=400, height=400)
skeleton_canvas.grid(row=0, column=1)

# Text outputs
character_label = tk.Label(root, text="Character: ", font=("Helvetica", 16), bg="white")
character_label.place(x=100, y=500)

sentence_label = tk.Label(root, text="Sentence: ", font=("Helvetica", 16), bg="white")
sentence_label.place(x=100, y=550)

# Buttons
btn_speak = tk.Button(root, text="Speak", font=("Helvetica", 14))
btn_speak.place(x=1000, y=500)

btn_clear = tk.Button(root, text="Clear", font=("Helvetica", 14))
btn_clear.place(x=1100, y=500)

# Suggestions (will update dynamically)
suggestion_label = tk.Label(root, text="Suggestions:", font=("Helvetica", 14), fg="red", bg="white")
suggestion_label.place(x=100, y=600)

# Placeholder webcam feed
cap = cv2.VideoCapture(0)

def update_webcam():
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        webcam_label.imgtk = imgtk
        webcam_label.configure(image=imgtk)
    webcam_label.after(10, update_webcam)

update_webcam()
root.mainloop()
cap.release()
cv2.destroyAllWindows()
