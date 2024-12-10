import os
import numpy as np


def load_data(action_path):
    ground_truth_path = os.path.join(action_path, "ground_truth.npy")
    mmwave_dir = os.path.join(action_path, "mmWave")

    ground_truth = np.load(ground_truth_path)
    mmwave_frames = []
    for frame_file in sorted(os.listdir(mmwave_dir)):
        if frame_file.endswith(".bin"):
            frame_path = os.path.join(mmwave_dir, frame_file)
            frame_data = np.fromfile(frame_path, dtype=np.float32).reshape(-1, 5)
            mmwave_frames.append(frame_data)
    return ground_truth, mmwave_frames


def preprocess_data(mmwave_frames, intensity_threshold=0.1, max_value=100):
    preprocessed_frames = []
    for frame in mmwave_frames:
        # Filter points by intensity
        frame = frame[frame[:, 4] > intensity_threshold]

        # Skip empty frames
        if frame.shape[0] == 0:
            continue

        # Clamp extreme values
        frame[:, :3] = np.clip(frame[:, :3], -max_value, max_value)

        # Normalize spatial coordinates safely
        frame[:, :3] = (frame[:, :3] - np.mean(frame[:, :3], axis=0)) / (np.std(frame[:, :3], axis=0) + 1e-8)
        frame[:, :3] = np.nan_to_num(frame[:, :3])  # Replace NaN/Inf with 0
        preprocessed_frames.append(frame)
    return preprocessed_frames


def create_mfpc(mmwave_frames, window_size=5, max_points=1024):
    mfpc = []
    for i in range(len(mmwave_frames) - window_size + 1):
        combined = np.vstack(mmwave_frames[i:i + window_size])

        # Pad or truncate to max_points
        if combined.shape[0] > max_points:
            combined = combined[:max_points]
        elif combined.shape[0] < max_points:
            padding = np.zeros((max_points - combined.shape[0], combined.shape[1]))
            combined = np.vstack((combined, padding))

        mfpc.append(combined)
    return mfpc
