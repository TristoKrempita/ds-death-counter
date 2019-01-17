import cv2
import datetime
import argparse
import os

now = datetime.datetime.now()

death_frame_count = 0
frame_count = 0
skip_frames = 0
threshold = .2

if not os.path.exists("frames"):
    os.makedirs("frames")

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to our file")
args = vars(ap.parse_args())

vidcap = cv2.VideoCapture(args["video"])
success,image = vidcap.read()

is_found = False
was_found = False

template = cv2.imread('youdied.png')
w, h = template.shape[:-1]
frame_name = f"frames/frame.jpg"
while success:

    skip_frames += 1
    #read frames  
    success = vidcap.grab()

    frame_count += 1
    #look at every 51st frame
    if skip_frames < 50: 
        continue
    success,image = vidcap.read()
    print(f'Read a new frame: {frame_count}')#, success)
    skip_frames = 0

    #finding "you died" text
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    print("Max Val: ", max_val)

    is_found = max_val >= threshold
    if is_found:
        prev_frame = image

    #shows image of death with frame number as name
    if was_found and not(is_found):
        death_frame_count += 1
        cv2.imwrite(f"frames/frame{frame_count}.jpg", prev_frame)

    was_found = is_found

print('='*50)
secs = (datetime.datetime.now()-now).seconds
print("Elapsed time: %sh:%sm:%ss" %((secs//3600)%24, (secs//60)%60, secs%60))
print("Deaths registered: ", death_frame_count)