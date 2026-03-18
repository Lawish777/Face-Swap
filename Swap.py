import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# ========= CHANGE THESE =========
SOURCE_IMAGE = "emma.jpg"   # Your face
TARGET_VIDEO = "target.mp4"   # Video to swap into
OUTPUT_VIDEO = "output.mp4"
# ================================c:\Users\ACER\Documents\Captura\target.mp4

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

def get_landmarks(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None
    h, w = img.shape[:2]
    return np.array([
        (int(p.x * w), int(p.y * h))
        for p in result.multi_face_landmarks[0].landmark
    ])

# Load source face
source_img = cv2.imread(SOURCE_IMAGE)
source_points = get_landmarks(source_img)

if source_points is None:
    raise RuntimeError("❌ No face detected in source image")

# Video setup
cap = cv2.VideoCapture(TARGET_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

print("🎬 Swapping faces in video...")

for _ in tqdm(range(total_frames)):
    ret, frame = cap.read()
    if not ret:
        break

    target_points = get_landmarks(frame)
    if target_points is None:
        out.write(frame)
        continue

    # Estimate transform
    M, _ = cv2.estimateAffinePartial2D(source_points, target_points)
    warped_source = cv2.warpAffine(source_img, M, (w, h))

    # Mask
    hull = cv2.convexHull(target_points)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    x, y, ww, hh = cv2.boundingRect(hull)
    center = (x + ww // 2, y + hh // 2)

    # Seamless clone
    result = cv2.seamlessClone(
        warped_source,
        frame,
        mask,
        center,
        cv2.NORMAL_CLONE
    )

    out.write(result)

cap.release()
out.release()

print("✅ DONE! Video saved as:", OUTPUT_VIDEO)
