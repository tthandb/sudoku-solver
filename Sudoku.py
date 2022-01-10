import cv2
import numpy as np
from PIL import Image, ImageTk


class Sudoku:
    def __init__(self, stage):
        self.stage = stage
        self.track_empty = np.ones((9, 9), dtype=int)
        self.initial_stage = stage

    def print_state(self):
        for row in range(9):
            print(self.stage[row])

    def get_empty_location(self):
        for row in range(9):
            for col in range(9):
                if self.stage[row][col] == 0:
                    self.track_empty[row][col] = 0
                    return row, col
        return -1, -1

    def check_row(self, row, number):
        for col in range(9):
            if self.stage[row][col] == number:
                return True
        return False

    def check_col(self, col, number):
        for row in range(9):
            if self.stage[row][col] == number:
                return True
        return False

    def check_minibox(self, row, col, number):
        for i in range(3):
            for j in range(3):
                if self.stage[i + row][j + col] == number:
                    return True
        return False

    def is_safe(self, row, col, number):
        return not self.check_col(col, number) and not self.check_row(row, number) and not self.check_minibox(
            row - row % 3, col - col % 3, number)

    def solve(self):
        r, c = self.get_empty_location()
        if r == -1:
            return True

        for number in range(1, 10):
            if self.is_safe(r, c, number):
                self.stage[r][c] = number
                if self.solve():
                    return True
                self.stage[r][c] = 0

        return False

    def state_to_imagetk(self):
        r = [24, 90, 156, 223, 289, 355, 422, 488, 554]
        c = [45, 111, 177, 244, 310, 376, 443, 509, 575]
        img = cv2.imread('assets/template.jpg')
        for j in range(9):
            for i in range(9):
                color = (63, 81, 181) if self.track_empty[j][i] == 0 else (0, 0, 0)
                cv2.putText(img, str(self.stage[j][i]), (r[i], c[j]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        img = Image.fromarray(img)
        return ImageTk.PhotoImage(img)

    def is_legal(self):
        check = True
        for i in range(9):
            if check_duplicate([val for val in self.stage[i] if val != 0]):
                return False
            col = [self.stage[j][i] for j in range(9) if self.stage[j][i] != 0]
            if check_duplicate(col):
                return False
        for i in range(3):
            for j in range(3):
                if check_duplicate(self.get_mini_box(i, j)):
                    return False
        return True

    def get_mini_box(self, r, c):
        minibox = []
        for i in range(3):
            for j in range(3):
                if self.stage[r * 3 + i][c * 3 + j] != 0:
                    minibox.append(self.stage[r * 3 + i][c * 3 + j])
        return minibox

    def no_solution_imagetk(self):
        img = cv2.imread('assets/no_solution.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        return ImageTk.PhotoImage(img)

    def init_imagetk(self):
        r = [24, 90, 156, 223, 289, 355, 422, 488, 554]
        c = [45, 111, 177, 244, 310, 376, 443, 509, 575]
        img = cv2.imread('assets/template.jpg')
        for j in range(9):
            for i in range(9):
                # color = (63, 81, 181) if self.track_empty[j][i] == 0 else (0, 0, 0)
                if self.initial_stage[j][i]:
                    cv2.putText(img, str(self.initial_stage[j][i]), (r[i], c[j]),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
        img = Image.fromarray(img)
        return ImageTk.PhotoImage(img)


def cant_detect_imagetk():
    img = cv2.imread('assets/cant_detect.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return ImageTk.PhotoImage(img)


def check_duplicate(ls):
    if len(ls) == len(set(ls)):
        return False
    return True

