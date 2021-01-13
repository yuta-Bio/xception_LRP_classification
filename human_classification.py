import glob
import sys
import tkinter
import os
import numpy as np
import cv2
import pandas as pd

def image_classification(path):
    ls_path = glob.glob(path + "/*.tif")
    df = pd.DataFrame(index=ls_path, columns=["class"])
    print(df)
    cv2.namedWindow("Image classification", cv2.WINDOW_KEEPRATIO)
    for path in ls_path:
        img = cv2.imread(str(path))
        cv2.imshow("Image classification", img)
        while True:
            key = cv2.waitKey()
            if key in list(range(48, 56)):
                df.at[str(path)] = key - 48
                break
            else:
                img = cv2.putText(img, "PLEASE ENTER 0~7", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    file_base = "image_classification"
    file_num = 1
    while True:
        file_name = file_base + str(file_num) + ".csv"
        if not os.path.isfile(file_name):
            break
        else:
            file_num += 1
    cv2.destroyAllWindows()
    df.to_csv(file_name)


def push_button(event):
    path = edit_box.get()
    image_classification(str(path[1:-1]))

if __name__ == "__main__":

    root = tkinter.Tk()
    edit_box = tkinter.Entry()
    edit_box.pack()
    push_bt = tkinter.Button(text='get Path')
    push_bt.bind('<Button-1>', push_button)
    push_bt.pack()
    root.mainloop()
    sys.exit()


