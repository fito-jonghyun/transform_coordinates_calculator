import os
import cv2
import glob
import argparse
import numpy as np
import pandas as pd
import tkinter as tk
import scipy.io as sio
import pickle
import copy


class TransformCalculator:
    def __init__(self, args):
        self.input = args.input
        self.start = args.start
        self.end = args.end
        self.width = args.width
        self.height = args.height
        self.grid_number = args.grid_number
        self.step_size = 0
        self.template = self.draw_pitch(img=None, width=self.width * 100, height=self.height * 100)

        self.is_editing = False
        self.is_finished = True

        self.define_world_coord_dict()

        self.pixel_coord = {}

        # self.pixel_coord = []
        self.world_coord_ID = []
        self.world_coord = []
        self.h_mat = []

        self.pixel_coord_list = {}

        self.estimated_pixel_coords = {}
        self.last_estimated_pixel_coords = {}
        self.next_estimated_pixel_coords = {}
        self.homographies = {}

        self.frame_indexes = []
        self.frame_width = None
        self.frame_height = None

        self.current_homography = None

    def load_results(self, filename=None):

        if filename is None:
            log_list = glob.glob('transform_mat_*.pickle')
            if len(log_list) == 0:
                print("No saved result.")
                return

            filename = log_list[0]

        with open(filename, "rb") as f:
            loaded_data = pickle.load(f)

        d = loaded_data['pixel_coord']
        self.pixel_coord_list = {key: d[key] for key in sorted(d.keys())}
        print(self.pixel_coord_list)


        print("Saved result loaded!")

    def interpolate_pixel_coords(self, current_index):
        saved_index_list = self.pixel_coord_list.keys()

        last_index = 0
        next_index = -1

        for index in saved_index_list:
            index = int(index)
            if index < current_index:
                last_index = index
            if index > current_index:
                next_index = index
                break

        self.world_coords_array = []
        self.pixel_coords_array = []

        if len(self.pixel_coord_list[last_index].keys()) >= 4:
            for key in self.pixel_coord_list[last_index].keys():
                self.pixel_coords_array.append(self.pixel_coord_list[last_index][key])
                self.world_coords_array.append((self.world_coord_dict[key][0] / self.width * self.template.shape[1],
                                                self.world_coord_dict[key][1] / self.height * self.template.shape[0]))

            self.last_homography, _ = cv2.findHomography(np.array(self.world_coords_array),
                                                            np.array(self.pixel_coords_array), cv2.RANSAC, 5.0)



        if next_index == -1:
            self.next_homography = self.last_homography
            next_index = current_index + 1
        else:
            self.world_coords_array = []
            self.pixel_coords_array = []
            if len(self.pixel_coord_list[next_index].keys()) >= 4:
                for key in self.pixel_coord_list[next_index].keys():
                    self.pixel_coords_array.append(self.pixel_coord_list[next_index][key])
                    self.world_coords_array.append((self.world_coord_dict[key][0] / self.width * self.template.shape[1],
                                                    self.world_coord_dict[key][1] / self.height * self.template.shape[0]))

                self.next_homography, _ = cv2.findHomography(np.array(self.world_coords_array),
                                                             np.array(self.pixel_coords_array), cv2.RANSAC, 5.0)
            else:
                self.next_homography = self.last_homography


        for index in range(last_index, next_index):
            self.estimated_pixel_coords[index] = {}

            for key in self.world_coord_dict.keys():
                self.last_estimated_pixel_coords[key] = self.coords_transform_world_to_pixel(
                    *self.world_coord_dict[key], self.last_homography)
                self.next_estimated_pixel_coords[key] = self.coords_transform_world_to_pixel(
                    *self.world_coord_dict[key],
                    self.next_homography)

                interpolate_weight = float(next_index - index) / (next_index - last_index)

                self.estimated_pixel_coords[index][key] = np.array([value * interpolate_weight for value in self.last_estimated_pixel_coords[key]]) + np.array([value  * (1 - interpolate_weight) for value in self.next_estimated_pixel_coords[key]])


        for index in range(last_index, next_index):
            print(index, "th Homography updated")
            self.homographies[index], _ = cv2.findHomography(np.array([self.world_coord_dict[key] for key in self.world_coord_dict.keys()]),
                               np.array([self.estimated_pixel_coords[index][key] for key in self.estimated_pixel_coords[index].keys()]),
                               cv2.RANSAC, 5.0)


        print("Homography interpolated from {} to {}".format(last_index, next_index))


    def main(self):
        self.cap = cv2.VideoCapture(self.input)
        self.TOTAL_LENGTH = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print(f"Video File Length: {self.TOTAL_LENGTH}\nSelected index: {self.start} ~ {self.end}")
        self.load_results()

        for stored_index in self.pixel_coord_list.keys():
            self.interpolate_pixel_coords(stored_index)

        self.cap.set(cv2.CAP_PROP_FPS, self.start)
        while self.cap.isOpened() and self.cap.get(cv2.CAP_PROP_POS_FRAMES) <= self.end:
            # try:
            self.frame_index = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            # if self.frame_index < self.frame_indexes[-1]:
            #    self.frame_index = self.frame_indexes[-1]

            ret, self.frame = self.cap.read()

            if self.frame_height is None:
                self.frame_height = self.frame.shape[0]
                self.frame_width = self.frame.shape[1]

            self.frame = cv2.putText(self.frame, f"FRAME: {self.frame_index}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                     2, (0, 0, 0), 3, cv2.LINE_AA)

            if self.frame_index in self.pixel_coord_list.keys():
                print("Current frame: {}, Stored coordinates: {}".format(self.frame_index, self.pixel_coord_list[self.frame_index]))

                frame_coord = self.pixel_coord_list[self.frame_index]
                for key in frame_coord.keys():
                    self.frame = cv2.putText(self.frame, f"{key}", (frame_coord[key][0], frame_coord[key][1]), cv2.FONT_HERSHEY_SIMPLEX,
                                         2, (0, 255, 0), 4, cv2.LINE_AA)
                    self.frame = cv2.circle(self.frame, (frame_coord[key][0], frame_coord[key][1]), 6, (0, 255, 0), 2)

                self.world_coords_array = []
                self.pixel_coords_array = []
                for key in self.pixel_coord_list[self.frame_index].keys():
                    self.pixel_coords_array.append(self.pixel_coord_list[self.frame_index][key])
                    self.world_coords_array.append((self.world_coord_dict[key][0] / self.width * self.template.shape[1], self.world_coord_dict[key][1]  / self.height * self.template.shape[0]))

                self.current_homography, _ = cv2.findHomography(np.array(self.world_coords_array),
                                                                np.array(self.pixel_coords_array), cv2.RANSAC, 5.0)

            elif len(self.pixel_coord_list.keys()) >= 2:
                self.interpolate_pixel_coords(self.frame_index)

                self.current_homography = self.homographies[self.frame_index]
                self.update_pixel_coords()

            if self.current_homography is not None:
                self.update_pixel_coords()

                transformed_pitch = cv2.warpPerspective(self.template, self.current_homography,
                                                        (self.frame_width, self.frame_height))

                img2gray = cv2.cvtColor(transformed_pitch, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)

                # Now black-out the area of logo in ROI
                img1_bg = cv2.bitwise_and(self.frame, self.frame, mask=mask_inv)

                # Take only region of logo from logo image.
                img2_fg = cv2.bitwise_and(transformed_pitch, transformed_pitch, mask=mask)

                self.frame = cv2.add(img1_bg, img2_fg)

            cv2.imshow('frame', self.frame)

            k = cv2.waitKey()
            if k == 32:  # space bar
                print("SPACE bar")
                self.frame = cv2.putText(self.frame, "EDITING MODE", (100, 170), cv2.FONT_HERSHEY_SIMPLEX,
                                         2, (255, 255, 255), 4, cv2.LINE_AA)

                cv2.imshow('frame', self.frame)

                self.is_editing = True
            elif k == ord('q'):
                self.is_finished = False
                break
            elif k == ord('s'):
                self.frame_indexes.append(self.frame_index)
                self.h_mat.append(self.h_mat[-1])
                self.move_next_frame(0)

            if self.is_editing:
                while True:
                    cv2.setMouseCallback('frame', self.mouse_event_handler)
                    k = cv2.waitKey()

                    if k == ord('q'):
                        self.is_finished = False
                        break
                    elif k == ord('b'):
                        self.move_next_frame(-5)
                        break
                    elif k == ord('j'):
                        frame_num_str = self.get_frame_num()
                        self.move_frame(int(frame_num_str))
                        break
                    elif k == ord('d'):
                        remove_list = self.get_world_coord()
                        remove_list = remove_list.split(", ")
                        for remove_key in remove_list:
                            del self.pixel_coord_list[self.frame_index][remove_key]

                        print("The following key points are removed:", remove_list)

                        self.world_coords_array = []
                        self.pixel_coords_array = []
                        for key in self.pixel_coord_list[self.frame_index].keys():
                            self.pixel_coords_array.append(self.pixel_coord_list[self.frame_index][key])
                            self.world_coords_array.append((self.world_coord_dict[key][0] / self.width * self.template.shape[1], self.world_coord_dict[key][1]  / self.height * self.template.shape[0]))

                        self.current_homography, _ = cv2.findHomography(np.array(self.world_coords_array),
                                                                        np.array(self.pixel_coords_array), cv2.RANSAC, 5.0)
                        break
                    elif k == ord('n') and (len(self.pixel_coord) >= 4 or len(self.pixel_coord_list[self.frame_index].keys()) >= 4):
                        self.frame_indexes.append(self.frame_index)
                        self.transform_world_ID_to_coord()

                        print(f"pixel coord: {self.pixel_coord}")

                        self.world_coords_array = []
                        self.pixel_coords_array = []
                        for key in self.pixel_coord.keys():
                            self.pixel_coords_array.append(self.pixel_coord[key])
                            self.world_coords_array.append((self.world_coord_dict[key][0] / self.width * self.template.shape[1], self.world_coord_dict[key][1]  / self.height * self.template.shape[0]))


                        self.current_homography, _ = cv2.findHomography(np.array(self.world_coords_array),
                                                                        np.array(self.pixel_coords_array), cv2.RANSAC, 5.0)
                        h_mat, _ = cv2.findHomography(np.array(self.pixel_coords_array), np.array(self.world_coords_array),
                                                      cv2.RANSAC, 5.0)
                        # print(f"homography matrix: {h_mat}")

                        self.h_mat.append(self.current_homography)

                        if self.pixel_coord is not None:
                            self.pixel_coord_list[self.frame_index] = copy.deepcopy(self.pixel_coord)

                        self.save_results(is_finished=False)

                        self.pixel_coord = {}
                        # self.world_coord = []
                        # self.world_coord_ID = []

                        self.move_next_frame(0)
                        self.is_editing = False
                        break

            if not self.cap.isOpened():
                self.save_results(is_finished=self.is_finished)
                cv2.destroyAllWindows()
                self.cap.release()
            # except:
            #    self.save_results(is_finished=False)
            #    print("ERROR!")nnn

    def draw_pitch(self, img, width, height):
        color = (0, 0, 255)
        scale = 5

        width = int(width / scale)
        height = int(height / scale)
        thickness = int(30 / scale)
        circle_radius = int(915 / scale)
        penalty_height = int(4030 / scale)
        penalty_width = int(1650 / scale)
        goal_height = int(1830 / scale)
        goal_width = int(550 / scale)
        penalty_spot_from_line = int(1100 / scale)

        if img == None:
            img = np.zeros((height, width, 3), np.uint8)

        cv2.rectangle(img, (0, 0), (width, height), color, thickness)
        cv2.rectangle(img, (0, 0), (width, height), color, thickness)
        cv2.circle(img, (int(width / 2), int(height / 2)), circle_radius, color, thickness)
        cv2.line(img, (int(width / 2), 0), (int(width / 2), height), color, thickness)
        cv2.rectangle(img, (0, int(height / 2) - int(penalty_height / 2)),
                      (penalty_width, int(height / 2) + int(penalty_height / 2)), color, thickness)
        cv2.rectangle(img, (width - penalty_width, int(height / 2) - int(penalty_height / 2)),
                      (width, int(height / 2) + int(penalty_height / 2)), color, thickness)
        cv2.rectangle(img, (0, int(height / 2) - int(goal_height / 2)),
                      (goal_width, int(height / 2) + int(goal_height / 2)), color, thickness)
        cv2.rectangle(img, (width - goal_width, int(height / 2) - int(goal_height / 2)),
                      (width, int(height / 2) + int(goal_height / 2)), color, thickness)
        cv2.circle(img, (penalty_spot_from_line, int(height / 2)), 10, color, thickness)
        cv2.circle(img, (width - penalty_spot_from_line, int(height / 2)), 10, color, thickness)

        return img

    def mouse_event_handler(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            #self.pixel_coord.append([x, y])
            self.draw_circle(x, y)
            cv2.imshow("frame", self.frame)
            world_coord_id = self.get_world_coord()

            self.pixel_coord[world_coord_id] = [x, y]

            # self.world_coord_ID.append(world_coord_id)

    def draw_circle(self, x, y):
        self.frame = cv2.circle(self.frame, (x, y), 4, (0, 0, 255), 2)

    # def remove_last_coord(self):
    #     if self.pixel_coord:
    #         self.draw_x_mark()
    #
    #         self.pixel_coord.pop()
    #         self.world_coord_ID.pop()
    #
    #         cv2.imshow("frame", self.frame)

    def draw_x_mark(self):
        self.frame = cv2.putText(self.frame, "X", (self.pixel_coord[-1][0] - 10, self.pixel_coord[-1][1] + 5),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    def move_next_frame(self, step_size=25):
        self.frame_number = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number + step_size)
        self.frame_index = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, self.frame = self.cap.read()

    def move_frame(self, dest_frame_num):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, dest_frame_num)
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

    def get_frame_num(self):
        root = tk.Tk()
        root.geometry("500x150")
        root.title('Frame num')

        ID = tk.StringVar(root)
        label = tk.Label(root, text='Frame num')
        label.pack()

        element = tk.Entry(root, textvariable=ID, width=20, fg="blue", bd=3, selectbackground='violet')
        element.pack()
        element.focus()
        button = tk.Button(root, text="Save & Quit", fg='White', bg='dark green', height=1, width=10,
                           command=root.destroy).pack()
        root.mainloop()

        return str(ID.get())

    def coords_transform_world_to_pixel(self, world_x, world_y, matrix):
        result_coords = (matrix @ np.array([world_x, world_y, 1]).reshape(3, 1)) / (matrix @ np.array([world_x, world_y, 1]).reshape(3, 1))[2]
        return result_coords[0], result_coords[1]

    def update_pixel_coords(self):
        for key in self.world_coord_dict.keys():
            self.estimated_pixel_coords[self.frame_index] = {}

            self.estimated_pixel_coords[self.frame_index][key] = self.coords_transform_world_to_pixel(self.world_coord_dict[key][0], self.world_coord_dict[key][1], self.current_homography)

        #print("test")

    def define_world_coord_dict(self):
        self.world_coord_dict = {
            "110": [0, 0],
            "120": [self.width / 2, 0],
            "130": [self.width, 0],

            "201": [0, self.height / 2 - 20.15],
            "202": [16.5, self.height / 2 - 20.15],
            "203": [0, self.height / 2 - 9.15],
            "204": [5.5, self.height / 2 - 9.15],
            "205": [16.5, self.height / 2 - 7.01],
            "206": [0, self.height / 2 - 3.65],
            "207": [11, self.height / 2],
            "208": [0, self.height / 2 + 3.65],
            "209": [0, self.height / 2 + 9.15],
            "210": [5.5, self.height / 2 + 9.15],
            "211": [16.5, self.height / 2 + 7.01],
            "212": [0, self.height / 2 + 20.15],
            "213": [16.5, self.height / 2 + 20.15],

            "301": [self.width / 2, self.height / 2 - 9.15],
            "302": [self.width / 2 - 9.15, self.height / 2],
            "303": [self.width / 2, self.height / 2],
            "304": [self.width / 2 + 9.15, self.height / 2],
            "305": [self.width / 2, self.height / 2 + 9.15],

            "401": [self.width - 16.5, self.height / 2 - 20.15],
            "402": [self.width, self.height / 2 - 9.15],
            "403": [self.width - 16.5, self.height / 2 - 7.01],
            "404": [self.width - 5.5, self.height / 2 - 9.15],
            "405": [self.width, self.height / 2 - 9.15],
            "406": [self.width, self.height / 2 - 3.65],
            "407": [self.width - 11, self.height / 2],
            "408": [self.width, self.height / 2 + 3.65],
            "409": [self.width - 16.5, self.height / 2 + 7.01],
            "410": [self.width - 5.5, self.height / 2 + 9.15],
            "411": [self.width, self.height / 2 + 9.15],
            "412": [self.width - 16.5, self.height / 2 + 20.15],
            "413": [self.width, self.height / 2 + 20.15],

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
                    self.world_coord_dict[str(header_list[j] + i + 1)] = [self.width / 2 + splited_width * i + 1,
                                                                          self.height]
        # print(self.world_coord_dict)

    def transform_world_ID_to_coord(self):
        for ID in self.world_coord_ID:
            self.world_coord.append([self.world_coord_dict[ID][0] * self.template.shape[1] / self.width,
                                     self.world_coord_dict[ID][1] * self.template.shape[0] / self.height])

    def save_results(self, is_finished):
        if is_finished:
            fname = f'transform_mat_{self.start}-{self.end}.pickle'
        else:
            fname = f'transform_mat_{self.start}-{self.frame_index}.pickle'

        fname = f'transform_mat_test.pickle'

        with open(fname, 'wb') as f:
            pickle.dump({"frame_index": self.frame_indexes,
                         "pixel_coord": self.pixel_coord_list}, f)

        if glob.glob("*.mat"):
            print("Transfrom Matrix was successfully saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str,
                        default='my_video.mp4', help='intput video file')
    parser.add_argument('--start', type=int,
                        default=0, help='start frame index')
    parser.add_argument('--end', type=int,
                        default=1000, help='end frame index')
    parser.add_argument('--width', type=int,
                        default=105, help='stadium width')
    parser.add_argument('--height', type=int,
                        default=68, help='stadium height')
    parser.add_argument('--grid-number', type=int,
                        default=10, help='the number of grass line in each half / TIP: count single side')

    args = parser.parse_args()

    transform_calc = TransformCalculator(args)
    transform_calc.main()
