import cv2
import sys
import mediapipe as mp

# ── MediaPipe solutions ────────────────────────────────────────────────────────
mp_pose    = mp.solutions.pose
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Hand landmark names for readable output
HAND_LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

# ── Load image ─────────────────────────────────────────────────────────────────
img_path = sys.argv[1]
image = cv2.imread(img_path)
h, w, _ = image.shape
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ── Run models ─────────────────────────────────────────────────────────────────
pose  = mp_pose.Pose(static_image_mode=True)
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

pose_results  = pose.process(rgb_image)
hands_results = hands.process(rgb_image)


# # ── Visualization ──────────────────────────────────────────────────────────────
# def visualize_keypoints(image, pose_results, hands_results, output_path="output.jpg"):
#     """
#     Draw pose and hand landmarks on the image and save to disk.

#     Args:
#         image:         Original BGR image (numpy array).
#         pose_results:  MediaPipe Pose results object.
#         hands_results: MediaPipe Hands results object.
#         output_path:   Path where the annotated image will be saved.
#     """
#     annotated = image.copy()

#     # — Pose landmarks —
#     if pose_results.pose_landmarks:
#         mp_drawing.draw_landmarks(
#             annotated,
#             pose_results.pose_landmarks,
#             mp_pose.POSE_CONNECTIONS,
#             landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
#         )
#         for i, lm in enumerate(pose_results.pose_landmarks.landmark):
#             cx, cy = int(lm.x * w), int(lm.y * h)
#             cv2.putText(
#                 annotated, str(i), (cx + 5, cy - 5),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA,
#             )

#     # — Hand landmarks —
#     if hands_results.multi_hand_landmarks:
#         for hand_landmarks, handedness in zip(
#             hands_results.multi_hand_landmarks,
#             hands_results.multi_handedness,
#         ):
#             label = handedness.classification[0].label  # "Left" or "Right"

#             # Skeleton + joints
#             mp_drawing.draw_landmarks(
#                 annotated,
#                 hand_landmarks,
#                 mp_hands.HAND_CONNECTIONS,
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style(),
#             )

#             # Landmark index labels (cyan for hands)
#             for i, lm in enumerate(hand_landmarks.landmark):
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 cv2.putText(
#                     annotated, str(i), (cx + 5, cy - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA,
#                 )

#             # Hand label at wrist position
#             wrist = hand_landmarks.landmark[0]
#             wx, wy = int(wrist.x * w), int(wrist.y * h)
#             cv2.putText(
#                 annotated, label, (wx - 20, wy - 15),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA,
#             )

#     cv2.imwrite(output_path, annotated)
#     print(f"Annotated image saved to: {output_path}")


# # ── Extract & print keypoints ──────────────────────────────────────────────────

# # Pose
# print("\n── POSE LANDMARKS (33) ──────────────────────────────")
# if pose_results.pose_landmarks:
#     for i, lm in enumerate(pose_results.pose_landmarks.landmark):
#         cx, cy = int(lm.x * w), int(lm.y * h)
#         print(f"  [{i:02d}] {mp_pose.PoseLandmark(i).name:<25} ({cx}, {cy})  visibility={lm.visibility:.2f}")
# else:
#     print("  No pose landmarks detected.")

# # Hands
# print("\n── HAND LANDMARKS (21 per hand) ─────────────────────")
if hands_results.multi_hand_landmarks:
    for hand_landmarks, handedness in zip(
        hands_results.multi_hand_landmarks,
        hands_results.multi_handedness,
    ):
        label = handedness.classification[0].label
        score = handedness.classification[0].score
        print(f"\n  {label} hand  (confidence={score:.2f})")
        for i, lm in enumerate(hand_landmarks.landmark):
            # cx, cy = int(lm.x * w), int(lm.y * h)
            # print(f"    [{i:02d}] {HAND_LANDMARK_NAMES[i]:<25} ({lm.x}, {lm.y})")
            print(f"{label} {score:.2f} ({lm.x}, {lm.y})")
# else:
#     print("  No hand landmarks detected.")

# # ── Visualize ──────────────────────────────────────────────────────────────────
# output_path = sys.argv[2] if len(sys.argv) > 2 else "output.jpg"
# visualize_keypoints(image, pose_results, hands_results, output_path)
