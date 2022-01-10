import cv2
import os
from PIL import Image, ImageTk
from tkinter import Tk, Label, Button, filedialog, CENTER

from Sudoku import *
from ImageProcess import ImageProcess


def select_image():
    global panelA, panelB

    path = filedialog.askopenfilename()
    base = os.path.basename(path)
    if len(path) > 0:
        image = cv2.imread(path)
        filename = os.path.splitext(base)[0]
        result = ImageProcess(image, filename)
        try:
            os.mkdir(filename)
        except OSError as error:
            pass
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = resize_to_imagetk_fit_panel(image)

        if panelA is None:
            panelA = Label(image=image)
            panelA.image = image
            panelA.pack(side="left", anchor=CENTER, padx=10, pady=10)
        else:
            panelA.configure(image=image)
            panelA.image = image

        result.preprocess_2()
        result.sort_title()
        try:
            stage = result.get_stage_num()
            print(stage)
        except:
            result_pic = cant_detect_imagetk()
            if panelB is None:
                panelB = Label(image=result_pic)
                panelB.image = result_pic
                panelB.pack(side="right", anchor=CENTER, padx=10, pady=10)
            else:
                panelB.configure(image=result_pic)
                panelB.image = result_pic
            return

        sudoku_game = Sudoku(stage)

        if sudoku_game.is_legal() and sudoku_game.solve():
            result_pic = sudoku_game.state_to_imagetk()
        else:
            result_pic = sudoku_game.init_imagetk()

        if panelB is None:
            panelB = Label(image=result_pic)
            panelB.image = result_pic
            panelB.pack(side="right", anchor=CENTER, padx=10, pady=10)
        else:
            panelB.configure(image=result_pic)
            panelB.image = result_pic


def resize_to_imagetk_fit_panel(image, arc_len=600):
    height, width, _ = image.shape
    newH, newW = (int(height * arc_len / width), arc_len) if width > height else (
        arc_len, int(width * arc_len / height))
    image = cv2.resize(image, (newW, newH))
    border_v, border_h = int((arc_len - newH) / 2), int((arc_len - newW) / 2)
    image = cv2.copyMakeBorder(image, border_v, border_v, border_h, border_h, cv2.BORDER_CONSTANT,
                               value=(255, 255, 255))
    image = Image.fromarray(image)
    return ImageTk.PhotoImage(image)


root = Tk(screenName='Sudoku solver', baseName='Sudoku solver')
root.title('Sudoku solver')
root.geometry('1250x670+50+50')
root.configure(background='white')

root.resizable(0, 0)
panelA = None
panelB = None

btn = Button(root, text="Select image", command=select_image, fg='black', bg='white')
btn.pack(side="bottom", anchor=CENTER, padx="10", pady="10")

root.mainloop()
