import cv2
import numpy as np
import math

def process_video(input_path, output_path, contrast_mode="linear"):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        progress = frame_idx / total_frames

        if contrast_mode == "linear":
            alpha = 0.8 + (1.5 - 0.8) * progress
        elif contrast_mode == "sine":
            alpha = 1.15 + 0.35 * math.sin(2 * math.pi * progress * 2)
        else:
            alpha = 1.0

        mean = np.mean(gray)
        adjusted = alpha * (gray.astype(np.float32) - mean) + mean
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

        out.write(adjusted)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Finished processing '{contrast_mode}' mode. Output saved to: {output_path}")

input_path = 'my_test_video.mp4'
process_video(input_path, 'KenCinco-FinalExer1/transformed_video_exer1_linear.mp4', contrast_mode='linear')
process_video(input_path, 'KenCinco-FinalExer1/transformed_video_exer1_pulsate.mp4', contrast_mode='sine')
