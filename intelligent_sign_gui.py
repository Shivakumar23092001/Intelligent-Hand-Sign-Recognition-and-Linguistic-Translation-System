import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pyttsx3
import os
import time
import sys

class SignLanguageApp:
    def __init__(self, root):
        print("[INFO] Initializing application...")

        self.root = root
        self.root.title("Intelligent Hand Sign Recognition")
        self.root.geometry("1300x800")

        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 150)

        try:
            self.model = load_model("landmark_model.h5")
            print("[OK] Model loaded.")
        except Exception as e:
            print("[ERROR] Model load failed:", e)
            sys.exit(1)

        self.labels_dict = {i: chr(65 + i) for i in range(26)}
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("[ERROR] Cannot access webcam.")
            sys.exit(1)
        print("[OK] Webcam initialized.")

        self.sentence = ""
        self.current_letter = ""
        self.current_word = ""
        self.last_action_time = time.time()

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils

        # ========== UI LAYOUT ==========
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack()

        self.camera_label = tk.Label(self.main_frame)
        self.camera_label.pack(side=tk.LEFT, padx=10, pady=10)

        self.side_canvas = tk.Canvas(self.main_frame, width=400, height=400, bg="white")
        self.side_canvas.pack(side=tk.RIGHT, padx=10, pady=10)

        self.char_label = tk.Label(self.root, text="Character: ", font=("Arial", 24), fg="blue")
        self.char_label.pack()

        self.sent_label = tk.Label(self.root, text="Sentence: ", font=("Arial", 20), fg="green")
        self.sent_label.pack()

        self.suggestion_label = tk.Label(self.root, text="Suggestions: ", font=("Arial", 16), fg="purple")
        self.suggestion_label.pack(pady=5)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=20)

        self.speak_btn = tk.Button(btn_frame, text="🔉 Speak", font=("Arial", 16), command=self.speak_sentence)
        self.speak_btn.pack(side=tk.LEFT, padx=20)

        self.clear_btn = tk.Button(btn_frame, text="❌ Clear", font=("Arial", 16), command=self.clear_sentence)
        self.clear_btn.pack(side=tk.LEFT, padx=20)

        self.update_frame()

    def speak_sentence(self):
        print("[INFO] Speaking:", self.sentence)
        self.engine.say(self.sentence)
        self.engine.runAndWait()

    def clear_sentence(self):
        print("[INFO] Clearing...")
        self.sentence = ""
        self.current_word = ""
        self.current_letter = ""
        self.char_label.config(text="Character: ")
        self.sent_label.config(text="Sentence: ")
        self.suggestion_label.config(text="Suggestions: ")
        self.side_canvas.delete("all")

    def get_finger_status(self, lm):
        status = []
        if lm[4].x < lm[3].x: status.append(1)
        else: status.append(0)
        tips = [8, 12, 16, 20]
        joints = [6, 10, 14, 18]
        for tip, joint in zip(tips, joints):
            if lm[tip].y < lm[joint].y: status.append(1)
            else: status.append(0)
        return status

    def suggest_words(self, prefix):
        suggestions = []
        try:
            with open("words.txt", "r") as f:
                for word in f:
                    word = word.strip().lower()
                    if word.startswith(prefix.lower()) and word not in suggestions:
                        suggestions.append(word)
                    if len(suggestions) >= 5:
                        break
        except:
            suggestions = ["(no words.txt found)"]
        return suggestions

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(30, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        white = 255 * np.ones((400, 400, 3), dtype=np.uint8)

        if result.multi_hand_landmarks:
            for hand_landmark in result.multi_hand_landmarks:
                h, w, _ = frame.shape
                landmarks = hand_landmark.landmark
                x_vals = [int(lm.x * w) for lm in landmarks]
                y_vals = [int(lm.y * h) for lm in landmarks]

                self.mp_draw.draw_landmarks(
                    frame, hand_landmark, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(0, 0, 0), thickness=2)
                )
                x_min, x_max = min(x_vals), max(x_vals)
                y_min, y_max = min(y_vals), max(y_vals)
                cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (255, 0, 0), 2)

                lm = hand_landmark.landmark
                finger_status = self.get_finger_status(lm)
                gesture = tuple(finger_status)
                now = time.time()

                if gesture == (1, 1, 1, 1, 1) and now - self.last_action_time > 2.5:
                    if self.current_letter:
                        self.sentence += self.current_letter
                        self.current_word += self.current_letter
                        self.char_label.config(text=f"Character: {self.current_letter}")
                        self.sent_label.config(text=f"Sentence: {self.sentence}")
                        suggestions = self.suggest_words(self.current_word)
                        self.suggestion_label.config(text="Suggestions: " + ", ".join(suggestions))
                        self.last_action_time = now
                    continue

                elif gesture == (1, 0, 0, 0, 0) and now - self.last_action_time > 2.5:
                    self.current_word = self.current_word[:-1]
                    self.sentence = self.sentence[:-1]
                    self.sent_label.config(text=f"Sentence: {self.sentence}")
                    suggestions = self.suggest_words(self.current_word)
                    self.suggestion_label.config(text="Suggestions: " + ", ".join(suggestions))
                    self.last_action_time = now
                    continue

                elif gesture == (0, 1, 1, 0, 0) and now - self.last_action_time > 2.5:
                    self.sentence += " "
                    self.current_word = ""
                    self.sent_label.config(text=f"Sentence: {self.sentence}")
                    self.suggestion_label.config(text="Suggestions: ")
                    self.last_action_time = now
                    continue

                # Draw skeleton on white
                self.mp_draw.draw_landmarks(white, hand_landmark, self.mp_hands.HAND_CONNECTIONS)

                # Predict
                resized = cv2.resize(white, (64, 64))
                normalized = resized / 255.0
                reshaped = np.expand_dims(normalized, axis=0)
                prediction = self.model.predict(reshaped)[0]
                class_index = np.argmax(prediction)
                confidence = prediction[class_index]

                if confidence > 0.6:
                    self.current_letter = self.labels_dict[class_index]
                    self.char_label.config(text=f"Character: {self.current_letter}")

        # Webcam feed
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.camera_label.imgtk = imgtk
        self.camera_label.configure(image=imgtk)

        # Side canvas feed
        pil_white = Image.fromarray(white)
        pil_white = pil_white.resize((400, 400))
        tkimg_white = ImageTk.PhotoImage(pil_white)
        self.side_canvas.imgtk = tkimg_white
        self.side_canvas.create_image(0, 0, anchor=tk.NW, image=tkimg_white)

        # Draw predicted letter
        if self.current_letter:
            self.side_canvas.create_text(
                200, 370,
                text=self.current_letter,
                font=("Arial", 36, "bold"),
                fill="blue"
            )

        # Exit on q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Exit key pressed.")
            self.cap.release()
            self.root.destroy()
            return

        self.root.after(30, self.update_frame)

if __name__ == '__main__':
    print("[INFO] Starting app...")
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()
