# MoveNet SinglePose (TFLite) — SavedModel-Compatible API

This project provides a **clean, commented Jupyter notebook** for running **MoveNet SinglePose Lightning (TFLite)** while keeping the **same API and output format** as the TF-Hub SavedModel version.

## Features
- **SavedModel-Compatible API** — `movenet(input_image)` returns `[1, 1, 17, 3]` keypoints + scores  
- **Auto-resizing** — feed any size RGB image; it’s resized inside the function  
- **Supports uint8, int32, float32 inputs** — dtype converted automatically  
- **Visualization helpers** — draw keypoints and skeleton connections  
- **Webcam & Video demos** — real-time pose detection from camera or video file  

---

## Requirements
- Python 3.8+
- TensorFlow 2.x  
- OpenCV (for image loading, resizing, and visualization)  

Install dependencies:
```bash
pip install tensorflow opencv-python
```

---

## Model Download
Download the **TFLite model** before running the notebook:
```bash
wget -O model.tflite "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite"
```
Place `model.tflite` in the same folder as the notebook.

---

## Usage

### 1. Run inference on an image
```python
import cv2
img_bgr = cv2.imread("person.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
kps = movenet(img_rgb)
print(kps.shape)  # (1, 1, 17, 3)
```

### 2. Visualize keypoints
```python
vis_rgb = draw_keypoints(img_rgb, kps, threshold=0.3)
cv2.imshow("Pose", cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
```
---

## Output Format
`movenet(input_image)` returns:
```
[1, 1, 17, 3]
```
Where for each keypoint:
- `y` = normalized vertical coordinate (0–1)
- `x` = normalized horizontal coordinate (0–1)
- `score` = confidence score (0–1)

Keypoints index mapping:
```
0: nose
1: left_eye
2: right_eye
3: left_ear
4: right_ear
5: left_shoulder
6: right_shoulder
7: left_elbow
8: right_elbow
9: left_wrist
10: right_wrist
11: left_hip
12: right_hip
13: left_knee
14: right_knee
15: left_ankle
16: right_ankle
```

---

## Notes
- This uses the **Lightning** variant of MoveNet — fastest single-person model.
- For higher accuracy, swap to **Thunder** TFLite model (input size 256) and adjust `INPUT_H`, `INPUT_W`.
- Ensure you have a GPU or optimized CPU build of TensorFlow for real-time speed.
