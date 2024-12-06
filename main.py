import cv2
import time

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

    cap.release()
    cv2.destroyAllWindows()


start_time = time.time()
track("vid/video5.mp4","MOSSE")
end_time = time.time()
time = end_time - start_time
print(f"Время выполнения: {time} секунд")