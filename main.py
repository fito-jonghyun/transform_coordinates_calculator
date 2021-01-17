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
        self.step_size = 25

        self.define_world_ID_coords()

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

            k = cv2.waitKey()
            if k == ord('q'):
                self.save_results(is_finished=False)
                cv2.destroyAllWindows()
                self.cap.release()
                break

            if k == ord('n'):
                self.frame_indexes.append(self.frame_index)
                h_mat = cv2.findHomography(np.array(self.pixel_coord), np.array(self.world_coord), cv2.RANSAC, 5.0)
                self.h_mat.append(h_mat)

                self.pixel_coord = []
                self.world_coord = []
                self.world_coord_ID = []

                self.move_next_frame()

        if self.frame_index == self.TOTAL_LENGTH:
            self.save_results(is_finished=True)
            cv2.destroyAllWindows()
            self.cap.release()

    def mouse_event_handler(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            self.pixel_coord.append([x, y])
            self.draw_circle()
            cv2.imshow("frame", self.frame)
            world_coord_id = self.get_world_coord()
            self.world_coord.append(world_coord_id)

    def draw_circle(self):
        self.frame = cv2.circle(self.frame, (self.pixel_coord[-1][0], self.pixel_coord[-1][1]), 4, (0, 0, 255), 2)

    def move_next_frame(self):
        self.frame_number = self.cap.set(cv2.CAP_PROP_POS_FRAMES)
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

        return ID.get()

    def define_world_ID_coords(self):
        self.world_coord_dict = {
            "101": np.array([0, 0]),
            "102": np.array([0, self.width / 2]),
            "103": np.array([0, self.width]),

            "201": np.array([0,     self.height / 2 - 20.16]),
            "202": np.array([16.5,  self.height / 2 - 20.16]),
            "203": np.array([0,     self.height / 2 - 9.16]),
            "204": np.array([5.5,   self.height / 2 - 9.16]),
            "205": np.array([16.5,  self.height / 2 - 7.01]),
            "206": np.array([0,     self.height / 2 - 3.66]),
            "207": np.array([11,    self.height / 2]),
            "208": np.array([0,     self.height / 2 + 3.16]),
            "209": np.array([0,     self.height / 2 + 9.16]),
            "210": np.array([5.5,   self.height / 2 + 9.16]),
            "211": np.array([16.5,  self.height / 2 + 7.01]),
            "212": np.array([0,     self.height / 2 + 20.16]),
            "213": np.array([16.5,  self.height / 2 + 20.16]),

            "301": np.array([self.width / 2,        self.height / 2 - 9.15]),
            "302": np.array([self.width / 2 - 9.15, self.height / 2]),
            "303": np.array([self.width / 2,        self.height / 2]),
            "304": np.array([self.width / 2 + 9.15, self.height / 2]),
            "305": np.array([self.width / 2,        self.height / 2 + 9.15]),

            "401": np.array([self.width - 16.5,     self.height / 2 - 20.16]),
            "402": np.array([self.width,            self.height / 2 - 9.16]),
            "403": np.array([self.width - 16.5,     self.height / 2 - 7.01]),
            "404": np.array([self.width - 5.5,      self.height / 2 - 9.16]),
            "405": np.array([self.width,            self.height / 2 - 9.16]),
            "406": np.array([self.width,            self.height / 2 - 3.66]),
            "407": np.array([self.width - 11,       self.height / 2]),
            "408": np.array([self.width,            self.height / 2 + 3.66]),
            "409": np.array([self.width - 16.5,     self.height / 2 + 7.01]),
            "410": np.array([self.width - 5.5,      self.height / 2 + 9.16]),
            "411": np.array([self.width,            self.height / 2 + 9.16]),
            "412": np.array([self.width - 16.5,     self.height / 2 + 20.16]),
            "413": np.array([self.width,            self.height / 2 + 20.16]),

            "501": np.array([0,                 self.height]),
            "502": np.array([self.width / 2,    self.height]),
            "503": np.array([self.width,        self.height]),
        }

    def transform_world_ID_to_coord(self):
        pass

    def save_results(self, is_finished):
        if is_finished:
            fname = f'Transform_mat_{self.start}-{self.end}.mat'
        else:
            fname = f'Transform_mat_{self.start}-{self.frame_index}.mat'

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

    args = parser.parse_args()

    transform_calc = TransformCalculator(args)
    transform_calc.main()