import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
from utils.face_recognition import recognize_face

def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        identity = recognize_face(file_path)
        label_result.config(text=f"Nhận diện: {identity}")
        img = Image.open(file_path)
        img = img.resize((250, 250), Image.ANTIALIAS)
        img_tk = ImageTk.PhotoImage(img)
        label_img.config(image=img_tk)
        label_img.image = img_tk

def launch_app():
    global label_img, label_result
    root = tk.Tk()
    root.title("Nhận diện khuôn mặt - Eigenfaces")

    Button(root, text="Chọn ảnh", command=open_file).pack()
    label_img = Label(root)
    label_img.pack()
    label_result = Label(root, text="Nhận diện: ")
    label_result.pack()

    root.mainloop()
