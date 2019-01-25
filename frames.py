import cv2
import datetime
import argparse
import os
import asyncio

#   Making CLI
if not os.path.exists("frames"):
    os.makedirs("frames")

t0 = datetime.datetime.now()
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
                help="path to our file")
args = vars(ap.parse_args())

threshold = .2
death_count = 0
was_found = False
template = cv2.imread('youdied.png')
vidcap = cv2.VideoCapture(args["video"])

loop = asyncio.get_event_loop()
frames_to_analyze = asyncio.Queue()


def main():

    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    tasks = []
    for _ in range(int(length / 50)):
        tasks.append(loop.create_task(read_frame(50, frames_to_analyze)))
        tasks.append(loop.create_task(analyze_frame(threshold, template, frames_to_analyze)))
    final_task = asyncio.gather(*tasks)
    loop.run_until_complete(final_task)

    dt = datetime.datetime.now() - t0
    print("App exiting, total time: {:,.2f} sec.".format(dt.total_seconds()))

    print("Synchronous version done in {:,.2f} seconds.".format(dt.total_seconds()))
    print(f"Deaths registered: {death_count}")


async def read_frame(frames, frames_to_analyze):
    global vidcap
    for _ in range(frames-1):
        if not (vidcap.grab()):
            # Stop loop:
            try:
                loop.stop()
                pending = asyncio.Task.all_tasks()
                loop.run_until_complete(asyncio.gather(*pending))
            except RuntimeError:
                print("Shutdown")
            print("Shutdown complete ...")
        await asyncio.sleep(0)
    else:
        current_frame = vidcap.read()[1]
    print("Read 50 frames")
    await frames_to_analyze.put(current_frame)


async def analyze_frame(threshold, template, frames_to_analyze):
    global vidcap
    global was_found
    global death_count
    frame = await frames_to_analyze.get()
    is_found = await processing_frame(frame)
    if was_found and not is_found:
        death_count += 1
        await writing_to_file(death_count, frame)
    was_found = is_found


async def processing_frame(frame):
    res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    max_val = cv2.minMaxLoc(res)[1]
    is_found = max_val >= threshold
    print(is_found)
    return is_found


async def writing_to_file(death_count, frame):
    cv2.imwrite(f"frames/frame{death_count}.jpg", frame)

if __name__ == '__main__':
    main()
