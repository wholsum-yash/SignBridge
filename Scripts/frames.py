import cv2


# extracting frames
def extract_frames(video_path, seq_len=32):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // seq_len)

    frames = []
    frame_idx = 0

    while len(frames) < seq_len:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        status, frame = cap.read()

        if not status:
            break
        frames.append(frame)
        frame_idx = frame_idx + step
    cap.release()

    # Padding if needed
    while len(frames) < seq_len:
        frames.append(frames[-1])

    return frames
