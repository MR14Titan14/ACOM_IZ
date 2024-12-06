import cv2
import time as t
import numpy as np

def track(video_path, tracker_type):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(f'res/{tracker_type}/{video_path.split("/")[1]}', fourcc, fps, (width, height))

    ret, frame = cap.read()
    if not ret:
        print("Не удалось прочитать первый кадр")
        exit()

    bbox = cv2.selectROI("Select", frame, False)
    cv2.destroyWindow("Select")

    start_time = t.time()
    if(tracker_type=="KCF"):
        tracker = cv2.TrackerKCF_create()
    if(tracker_type=="CSRT"):
        tracker = cv2.TrackerCSRT_create()
    if(tracker_type=="MOSSE"):
        tracker = cv2.legacy.TrackerMOSSE_create()

    tracker.init(frame, bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _, bbox = tracker.update(frame)

        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

        cv2.imshow(f"{tracker_type}", frame)
        output_video.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end_time = t.time()
    time = end_time - start_time
    print(f"Время выполнения: {time} секунд")
    cap.release()
    cv2.destroyAllWindows()
def custom_track(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(f'res/MS/{video_path.split("/")[1]}', fourcc, fps, (width, height))

    ret, frame = cap.read()

    if not ret:
        print("Не удалось прочитать первый кадр")
        exit()

    bbox = cv2.selectROI('Select', frame, False)
    cv2.destroyWindow("Select")

    start_time = t.time()
    x, y, w, h = bbox

    roi = frame[y:y + h, x:x + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gaus_roi=cv2.GaussianBlur(hsv_roi,(7,7),100)
    mask = cv2.inRange(gaus_roi, np.array((0., 60., 32.)),np.array((180., 255., 255.)))
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
    roi_hist = cv2.calcHist([gaus_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        bp = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        ret, track_window = cv2.meanShift(bp, bbox, term_crit)

        x, y, w, h = track_window
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
        cv2.imshow('MS', frame)

        output_video.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    end_time = t.time()
    time = end_time - start_time
    print(f"Время выполнения: {time} секунд")
    cap.release()
    cv2.destroyAllWindows()

# track("vid/video5.mp4","KCF")
custom_track("vid/video2.mp4")