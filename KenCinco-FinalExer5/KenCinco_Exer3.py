import cv2
import numpy as np

def apply_moving_blur(input_path, output_path_hard, output_path_blend, kernel_size=21, blur_region_width=100):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_hard = cv2.VideoWriter(output_path_hard, fourcc, fps, (width, height))
    out_blend = cv2.VideoWriter(output_path_blend, fourcc, fps, (width, height))

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        progress = frame_idx / total_frames
        cx = int(progress * width)

        half_w = blur_region_width // 2
        x_start = max(0, cx - half_w)
        x_end = min(width, cx + half_w)
        region_slice = slice(x_start, x_end)

        blurred = cv2.blur(frame, (kernel_size, kernel_size))

        # --- HARD CUT VERSION ---
        output_hard = frame.copy()
        output_hard[:, region_slice] = blurred[:, region_slice]
        out_hard.write(output_hard)

        # --- SMOOTH BLEND VERSION ---
        output_blend = frame.copy()
        alpha = np.zeros((height, x_end - x_start, 1), dtype=np.float32)
        for x in range(x_end - x_start):
            rel_x = x / (x_end - x_start)
            alpha[:, x, 0] = np.sin(rel_x * np.pi)  # sine from 0 to 1 to 0
        alpha = np.clip(alpha, 0, 1)
        alpha_3ch = np.repeat(alpha, 3, axis=2)

        blended = (blurred[:, region_slice] * alpha_3ch + frame[:, region_slice] * (1 - alpha_3ch)).astype(np.uint8)
        output_blend[:, region_slice] = blended
        out_blend.write(output_blend)

    cap.release()
    out_hard.release()
    out_blend.release()
    print(f"Hard-cut output saved to: {output_path_hard}")
    print(f"Smooth-blend output saved to: {output_path_blend}")


input_path = 'my_test_video.mp4'
output_path_hard = 'KenCinco-FinalExer2/transformed_video_exer2_hardcut.mp4'
output_path_blend = 'KenCinco-FinalExer2/transformed_video_exer2_blend.mp4'
apply_moving_blur(input_path, output_path_hard, output_path_blend)
