import cv2 as cv
import numpy as np
import time
import datetime
from PIL import Image, ImageDraw, ImageFont


class VideoState:
    def __init__(self, width, height):
        self.record_state = False
        self.record_time_count = 0
        self.record_start_time = 0
        self.frame_size = (width, height)
        self.line_color = (0, 0, 255)
        self.line_thickness = 5
        self.text_size = 4
        self.text_color = (0, 191, 255)
        self.text_name = '胡单春'
        self.text_id = '21921082'
        self.photo_path = 'logo.png'
        self.recording_path = 'recording.png'
        self.stop_recording_path = 'stop_record.png'
        self.time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    def update_time(self):
        self.time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    def update_record_state(self):
        if not self.record_state:
            self.record_start_time = datetime.datetime.now()
        else:
            self.record_time_count += (datetime.datetime.now() - self.record_start_time).seconds
        self.record_state = not self.record_state


class LineTrace:
    def __init__(self, start):
        self.points = [start]

    def add_point(self, point):
        self.points.append(point)


line_traces = []
line = None
draw_line_flag = False


def mouse_draw(event, x, y, flags, param):
    global draw_line_flag, line_traces, line
    if event == cv.EVENT_LBUTTONDOWN:
        draw_line_flag = True
        line = LineTrace((x, y))
        line_traces.append(line)
    elif event == cv.EVENT_MOUSEMOVE:
        if draw_line_flag:
            line.add_point((x, y))
    elif event == cv.EVENT_LBUTTONUP:
        draw_line_flag = False
        line = None
    else:
        pass


def draw_line_traces(frame, video_state):
    for trace in line_traces:
        for idx, point in enumerate(trace.points):
            # the first point of a line trace, should be a circle
            if idx == 0:
                draw_circle(frame, point, (0, 0, 255))
            else:
                draw_line(frame, trace.points[idx-1], point, (0, 0, 255), video_state.line_thickness)


def record_save_play_video():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Can't open the camera, Please check it!")
        exit(0)

    video_state = VideoState(int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))

    cv.namedWindow('image', 0)
    cv.setMouseCallback('image', mouse_draw)

    frame_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        key = cv.waitKey(100)
        # keyboard == space
        if key == 32:
            video_state.update_record_state()
        # keyboard == ESC
        if key == 27:
            break
        # draw line traces
        draw_line_traces(frame, video_state)
        # refresh the time showed
        video_state.update_time()
        # add some information about self
        text = video_state.text_name + "\n" + video_state.text_id + "\n" + video_state.time
        frame = add_text(frame, text, video_state.text_color, (1000, 600), 'right')
        # add picture as logo
        add_pic_logo(frame, video_state, 30)
        # frame to be stored is copy from frame
        store_frame = frame.copy()
        # add recording icon or stop-record icon to show whether to record or not
        add_record_icon(frame, video_state)
        frame = add_text(frame, "Esc键退出，Space键开始/暂停录制", (255, 0, 0), (400, 0))

        # record
        if video_state.record_state:
            frame_list.append(store_frame)
        # not record
        else:
            pass

        # show
        cv.imshow('image', frame)


    if video_state.record_time_count != 0:
        fps = len(frame_list) / video_state.record_time_count
    else:
        fps = 25

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter('test.mp4', fourcc, fps,
                         video_state.frame_size, True)
    for frame in frame_list:
        out.write(frame)

    # release all
    cap.release()
    out.release()
    cv.destroyAllWindows()

# draw line
def draw_line(img, start, end, color, thickness):
    cv.line(img, start, end, color, thickness)

# draw circle
def draw_circle(img, point, color):
    cv.circle(img, point, 1, color, -1)


def add_pic_logo(frame, video_state, scale_percent):
    piclogo = cv.imread(video_state.photo_path)
    width = int(piclogo.shape[1]*scale_percent/100)
    height = int(piclogo.shape[0]*scale_percent/100)
    piclogo = cv.resize(piclogo, (width, height))
    frame[0:height, video_state.frame_size[0]-width:] = piclogo


def add_record_icon(frame, video_state):
    if video_state.record_state:
        icon = cv.imread(video_state.recording_path)
    else:
        icon = cv.imread(video_state.stop_recording_path)

    roi = frame[20:icon.shape[0]+20, 20:icon.shape[1]+20]
    icongray = cv.cvtColor(icon, cv.COLOR_BGR2GRAY)
    ret, mask = cv.threshold(icongray, 0, 255, cv.THRESH_BINARY)
    mask_inv = cv.bitwise_not(mask)

    frame_bg = cv.bitwise_and(roi, roi, mask=mask_inv)
    icon_fg = cv.bitwise_and(icon, icon, mask=mask)
    dst = cv.add(frame_bg, icon_fg)

    frame[20:icon.shape[0]+20, 20:icon.shape[1]+20] = dst


def add_text(frame, text, color, pos, align='left'):
    if isinstance(frame, np.ndarray):
        frame = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(frame)
    font = ImageFont.truetype("./Arial Unicode.ttf", 30, encoding="utf-8")

    draw.text(pos, text, color, font, align)
    return cv.cvtColor(np.asarray(frame), cv.COLOR_RGB2BGR)


if __name__ == '__main__':
    record_save_play_video()
