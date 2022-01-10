import cv2
from PIL import Image, ImageTk
import numpy as np
from utils import get_digit
from functools import cmp_to_key


class ImageProcess:
    def __init__(self, image, name):
        self.name = name
        self.image = image
        self.list_titles = []
        self.img_state = []
        self.num_stage = np.zeros((9, 9), dtype=int)

    def process(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 50, 100)
        self.image = edged

    def resize(self, w=600, h=600):
        self.image = cv2.resize(self.image, (w, h))

    def image_to_imagetk_fit_panel(self, arc_len=600):
        height, width = self.image.shape
        newH, newW = (int(height * arc_len / width), arc_len) if width > height else (
            arc_len, int(width * arc_len / height))
        image = cv2.resize(self.image, (newW, newH))
        border_v, border_h = int((arc_len - newH) / 2), int((arc_len - newW) / 2)
        image = cv2.copyMakeBorder(image, border_v, border_v, border_h, border_h, cv2.BORDER_CONSTANT,
                                   value=(255, 255, 255))
        image = Image.fromarray(image)
        return ImageTk.PhotoImage(image)
  
    def sort_title(self):
        self.list_titles = sorted(self.list_titles, key=cmp_to_key(cmp), reverse=True)
        self.img_state = [self.list_titles[0:9],
                          self.list_titles[9:18],
                          self.list_titles[18:27],
                          self.list_titles[27:36],
                          self.list_titles[36:45],
                          self.list_titles[45:54],
                          self.list_titles[54:63],
                          self.list_titles[63:72],
                          self.list_titles[72:81]]

    def get_stage_num(self):
        for i in range(9):
            for j in range(9):
                num = get_digit(self.img_state[i][j][1], self.name, str(i + 1) + '_' + str(j+1))
                if num:
                    self.num_stage[i][j] = int(num)
        return self.num_stage

    def preprocess_2(self):
        # Resize img to reduce time
        height, width, _ = self.image.shape
        newH, newW = int(height * 1500 / width), 1500
        self.resize(newW, newH)

        image0 = self.image
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        dilate = cv2.dilate(self.image, np.ones((7, 7), np.uint8))
        diff_img = 255 - cv2.absdiff(self.image, cv2.medianBlur(dilate, 21))

        high_thresh, _ = cv2.threshold(diff_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        low_thresh = 0.5 * high_thresh
        threshold_img = cv2.adaptiveThreshold(diff_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 7)
        canny_img = cv2.Canny(cv2.medianBlur(threshold_img, 3), low_thresh, high_thresh)
        image4 = cv2.dilate(canny_img, cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)), iterations=3)

        image8 = diff_img.copy()
        contours, hierachy = cv2.findContours(image4, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        c = [[i, contours[i], cv2.contourArea(contours[i])] for i in range(len(contours))]
        maxCon = max(c, key=lambda c: c[2])

        core = cv2.moments(maxCon[1])
        cX = int(core["m10"] / core["m00"])
        cY = int(core["m01"] / core["m00"])
        peri = cv2.arcLength(maxCon[1], True)
        cnx = cv2.convexHull(maxCon[1], returnPoints=True)
        approx = cv2.approxPolyDP(cnx, 0.05 * peri, True)
        min_approx = min([[k, approx[k]] for k in range(4)], key=lambda x: x[1][0][0] + x[1][0][1])
        new_approx = approx.copy()
        for k in range(4):
            new_approx[k] = approx[(k + min_approx[0]) % 4]
        approx = new_approx

        after = np.array([[cX - 300, cY - 300], [cX + 300, cY - 300], [cX + 300, cY + 300], [cX - 300, cY + 300]],
                         dtype="float32")
        approx = np.array([approx[0][0], approx[1][0], approx[2][0], approx[3][0]], dtype="float32")  # 4 đỉnh ban đầu

        for k in range(4):
            if approx[k][0] < cX:
                approx[k][0] -= 30
            else:
                approx[k][0] += 30
            if approx[k][1] < cY:
                approx[k][1] -= 30
            else:
                approx[k][1] += 30

        perspect = cv2.getPerspectiveTransform(approx, after)
        res0 = cv2.warpPerspective(image8, perspect, (newW, newH))
        res4 = cv2.warpPerspective(image4, perspect, (newW, newH))
        res0 = res0[cY - 300:cY + 300, cX - 300:cX + 300]
        res4 = res4[cY - 300:cY + 300, cX - 300:cX + 300]

        contours, hierachy = cv2.findContours(res4.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        c = [[i, contours[i], cv2.contourArea(contours[i])] for i in range(len(contours))]
        maxCon = max(c, key=lambda c: c[2])

        child = hierachy[0][maxCon[0]][2]
        inner = []
        while child != -1:
            inner.append(child)
            child = hierachy[0][child][0]

        # Với từng ô con: tìm 4 đỉnh, tâm, sau đó xoay hình ban đầu để ô con đó về hình vuông
        for j in range(len(inner)):
            if cv2.contourArea(contours[inner[j]]) > maxCon[2] / 400:
                image8 = res0.copy()
                i = inner[j]

                # tìm tâm: (cX,cY) là tọa độ
                core = cv2.moments(contours[i])
                cX = int(core["m10"] / core["m00"])
                cY = int(core["m01"] / core["m00"])
                # print(j, i, cX, cY)

                peri = cv2.arcLength(contours[i], True)
                cnx = cv2.convexHull(contours[i], returnPoints=True)

                approx = cv2.approxPolyDP(cnx, 0.03 * peri, True)
                min_approx = min([[k, approx[k]] for k in range(4)], key=lambda x: x[1][0][0] + x[1][0][1])
                new_approx = approx.copy()
                for k in range(4):
                    new_approx[k] = approx[(k + min_approx[0]) % 4]
                approx = new_approx

                # biến đổi hình thái sao cho ô sudoku đang xét vê hình vuông
                after = np.array([[cX - 45, cY - 45], [cX + 45, cY - 45], [cX + 45, cY + 45], [cX - 45, cY + 45]],
                                 dtype="float32")
                approx = np.array([approx[0][0], approx[1][0], approx[2][0], approx[3][0]],
                                  dtype="float32")
                perspect = cv2.getPerspectiveTransform(approx, after)

                res = cv2.warpPerspective(image8, perspect, (newW, newH))
                # res = res[cY - 45:cY + 45, cX - 45:cX + 45]
                res = res[max(0, cY - 45):min(newW, cY + 45), max(0, cX - 45):min(newH, cX + 45)]
                _, res = cv2.threshold(res.copy(), 225, 255, cv2.THRESH_BINARY)
                self.list_titles.append([(cX, cY), res])


def cmp(x, y):
    if abs(x[0][1] - y[0][1]) < 30:
        return -1 if x[0][0] > y[0][0] else 1
    return -1 if x[0][1] > y[0][1] else 1

