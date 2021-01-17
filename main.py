import os
import cv2
import glob
import argparse
import numpy as np
import pandas as pd
import tkinter as tk
import scipy.io as sio

class TransformCalculator:
    def __init__(self, args):
        self.input = args.input
        self.start = args.start
        self.end   = args.end
        self.width = args.width
        self.height = args.height
        self.grid_number = args.grid_number
        self.step_size = 25

        self.is_finished = True

        self.define_world_coord_dict()

        self.pixel_coord = []
        self.world_coord_ID = []
        self.world_coord = []
        self.h_mat = []
        self.frame_indexes = []

    def main(self):
        self.cap = cv2.VideoCapture(self.input)
        self.TOTAL_LENGTH = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print(f"Video File Length: {self.TOTAL_LENGTH}\nSelected index: {self.start} ~ {self.end}")

        self.cap.set(cv2.CAP_PROP_FPS, self.start)
        while self.cap.isOpened() and self.cap.get(cv2.CAP_PROP_POS_FRAMES) <= self.end:
            self.frame_index = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, self.frame = self.cap.read()
            self.frame = cv2.putText(self.frame, f"FRAME: {self.frame_index}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                     2, (0, 0, 0), 3, cv2.LINE_AA)

            cv2.imshow('frame', self.frame)
            cv2.setMouseCallback('frame', self.mouse_event_handler)

            k = cv2.waitKey(0) & 0xFF

            if k == ord('q'):
                self.is_finished = False
                break
            elif k == ord('r'):
                self.remove_last_coord()

            if k == ord('n'):
                self.frame_indexes.append(self.frame_index)
                self.transform_world_ID_to_coord()

                print(f"pixel coord: {self.pixel_coord} world coord: {self.world_coord}")

                h_mat = cv2.findHomography(np.array(self.pixel_coord), np.array(self.world_coord), cv2.RANSAC, 5.0)
                # print(f"homography matrix: {h_mat}")

                self.h_mat.append(h_mat[0])

                self.pixel_coord = []
                self.world_coord = []
                self.world_coord_ID = []

                self.move_next_frame()
                self.condition_fullfillment = False

        if not self.cap.isOpened():
            self.save_results(is_finished=self.is_finished)
            cv2.destroyAllWindows()
            self.cap.release()

    def mouse_event_handler(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            self.pixel_coord.append([x, y])
            self.draw_circle()
            cv2.imshow("frame", self.frame)
            world_coord_id = self.get_world_coord()
            self.world_coord_ID.append(world_coord_id)

    def draw_circle(self):
        self.frame = cv2.circle(self.frame, (self.pixel_coord[-1][0], self.pixel_coord[-1][1]), 4, (0, 0, 255), 2)

    def remove_last_coord(self):
        if self.pixel_coord:
            self.draw_x_mark()

            self.pixel_coord.pop()
            self.world_coord_ID.pop()

            cv2.imshow("frame", self.frame)

    def draw_x_mark(self):
        self.frame = cv2.putText(self.frame, "X", (self.pixel_coord[-1][0], self.pixel_coord[-1][1]),
                                 cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)


    def move_next_frame(self):
        self.frame_number = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number + self.step_size)
        self.frame_index = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, self.frame = self.cap.read()

    def get_world_coord(self):
        root = tk.Tk()
        root.geometry("500x150")
        root.title('World coord ID')

        ID = tk.StringVar(root)
        label = tk.Label(root, text='World coord ID')
        label.pack()

        element = tk.Entry(root, textvariable=ID, width=20, fg="blue", bd=3, selectbackground='violet')
        element.pack()
        element.focus()
        button = tk.Button(root, text="Save & Quit", fg='White', bg='dark green', height=1, width=10,
                           command=root.destroy).pack()
        root.mainloop()

        return str(ID.get())

    def define_world_coord_dict(self):
        self.world_coord_dict = {
            "110": [0, 0],
            "120": [0, self.width / 2],
            "130": [0, self.width],

            "201": [0, self.height / 2 - 20.16],
            "202": [16.5, self.height / 2 - 20.16],
            "203": [0, self.height / 2 - 9.16],
            "204": [5.5, self.height / 2 - 9.16],
            "205": [16.5, self.height / 2 - 7.01],
            "206": [0, self.height / 2 - 3.66],
            "207": [11, self.height / 2],
            "208": [0, self.height / 2 + 3.16],
            "209": [0, self.height / 2 + 9.16],
            "210": [5.5, self.height / 2 + 9.16],
            "211": [16.5, self.height / 2 + 7.01],
            "212": [0, self.height / 2 + 20.16],
            "213": [16.5, self.height / 2 + 20.16],

            "301": [self.width / 2, self.height / 2 - 9.15],
            "302": [self.width / 2 - 9.15, self.height / 2],
            "303": [self.width / 2, self.height / 2],
            "304": [self.width / 2 + 9.15, self.height / 2],
            "305": [self.width / 2, self.height / 2 + 9.15],

            "401": [self.width - 16.5, self.height / 2 - 20.16],
            "402": [self.width, self.height / 2 - 9.16],
            "403": [self.width - 16.5, self.height / 2 - 7.01],
            "404": [self.width - 5.5, self.height / 2 - 9.16],
            "405": [self.width, self.height / 2 - 9.16],
            "406": [self.width, self.height / 2 - 3.66],
            "407": [self.width - 11, self.height / 2],
            "408": [self.width, self.height / 2 + 3.66],
            "409": [self.width - 16.5, self.height / 2 + 7.01],
            "410": [self.width - 5.5, self.height / 2 + 9.16],
            "411": [self.width, self.height / 2 + 9.16],
            "412": [self.width - 16.5, self.height / 2 + 20.16],
            "413": [self.width, self.height / 2 + 20.16],

            "510": [0, self.height],
            "520": [self.width / 2, self.height],
            "530": [self.width, self.height],
        }

        header_list = [110, 120, 510, 520]
        splited_width = self.width / 2 / self.grid_number

        for i in range(self.grid_number - 1):
            for j in range(4):
                if j == 0:
                    self.world_coord_dict[str(header_list[j] + i + 1)] = [splited_width * i + 1, 0]
                elif j == 1:
                    self.world_coord_dict[str(header_list[j] + i + 1)] = [self.width / 2 + splited_width * i + 1, 0]
                elif j == 2:
                    self.world_coord_dict[str(header_list[j] + i + 1)] = [splited_width * i + 1, self.height]
                elif j == 3:
                    self.world_coord_dict[str(header_list[j] + i + 1)] = [self.width / 2 + splited_width * i + 1, self.height]
        # print(self.world_coord_dict)

    def transform_world_ID_to_coord(self):
        for ID in self.world_coord_ID:
            self.world_coord.append(self.world_coord_dict[ID])

    def save_results(self, is_finished):
        if is_finished:
            fname = f'transform_mat_{self.start}-{self.end}.mat'
        else:
            fname = f'transform_mat_{self.start}-{self.frame_index}.mat'

        sio.savemat(fname, {"frame_index": self.frame_indexes, "h_mat": self.h_mat})

        if glob.glob("*.mat"):
            print("Transfrom Matrix was successfully saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, required=True,
                        default='my_video.mp4', help='intput video file')
    parser.add_argument('--start', type=int, required=True,
                        default=0, help='start frame index')
    parser.add_argument('--end', type=int, required=True,
                        default=9999, help='end frame index')
    parser.add_argument('--width', type=int, required=True,
                        default=105, help='stadium width')
    parser.add_argument('--height', type=int, required=True,
                        default=68, help='stadium height')
    parser.add_argument('--grid-number', type=int, required=True,
                        default=10, help='the number of grass line in each half / TIP: count single side')

    args = parser.parse_args()

    transform_calc = TransformCalculator(args)
    transform_calc.main()