"""
MemeCV - Detecção de expressões faciais e gestos de mão com memes
=================================================================
Requer Python 3.8+ com opencv-python e mediapipe==0.10.32 instalados.

Uso:
    python main.py

Tecla ESC: encerra o programa.
"""

import cv2
import numpy as np
from collections import deque
import os
import sys

# ---------------------------------------------------------------------------
# Configuração do MediaPipe (Importação Blindada para Windows)
# ---------------------------------------------------------------------------
# No Windows, se mp.solutions falhar, importamos diretamente do núcleo.
try:
    import mediapipe as mp
    # Tentativa de importação direta dos módulos internos
    from mediapipe.python.solutions import face_mesh as mp_face_mesh
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
except (AttributeError, ImportError):
    try:
        # Segunda tentativa: via mp.solutions (padrão)
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        mp_hands     = mp.solutions.hands
        mp_drawing   = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
    except Exception as e:
        print("=" * 60)
        print(f"ERRO CRÍTICO: O MediaPipe não pôde ser carregado corretamente: {e}")
        print("\nSiga estes passos para resolver no Windows:")
        print("1. Delete a pasta 'venv' do seu projeto.")
        print("2. No terminal, rode: pip uninstall mediapipe opencv-python opencv-contrib-python -y")
        print("3. Depois rode: pip install mediapipe==0.10.32 opencv-python")
        print("=" * 60)
        sys.exit(1)

# ---------------------------------------------------------------------------
# Caminhos dos memes
# ---------------------------------------------------------------------------
MEME_DIR = os.path.join(os.path.dirname(__file__), "assets", "new")

MEME_PATHS = {
    "smile_peace":       os.path.join(MEME_DIR, "109fb257daabe2f3db63bd7bc1944934.jpg"),
    "thinking_default":  os.path.join(MEME_DIR, "maxresdefault.jpg"),
    "thumbs_up":         os.path.join(MEME_DIR, "7dc6efb0fe7548ae00dd6143e739f630.jpg"),
    "timeout":           os.path.join(MEME_DIR, "bc3d38ffc8a2e9a574bb54d3bffa5445.jpg"),
}

# ---------------------------------------------------------------------------
# Estabilização de gestos (7 frames consecutivos)
# ---------------------------------------------------------------------------
HISTORY_LEN = 7
_gesture_history = deque(maxlen=HISTORY_LEN)

# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def load_meme(key, path):
    if os.path.exists(path):
        img = cv2.imread(path)
        if img is not None:
            return img
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, f"[{key}]", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 120, 255), 2)
    cv2.putText(img, "Adicione o meme em assets/new/", (40, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 255), 2)
    cv2.putText(img, os.path.basename(path), (40, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 255), 2)
    return img

def resize_to_fit(img, width=640, height=480):
    h, w = img.shape[:2]
    scale = min(width / w, height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y_off = (height - new_h) // 2
    x_off = (width  - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas

def stable_gesture(current):
    _gesture_history.append(current)
    if len(_gesture_history) == HISTORY_LEN and all(g == current for g in _gesture_history):
        return current
    return None

# ---------------------------------------------------------------------------
# Detecção de expressões faciais
# ---------------------------------------------------------------------------

def detect_face_expression(face_landmarks):
    lm = face_landmarks.landmark
    # Sorriso largo + boca aberta
    mouth_width = abs(lm[291].x - lm[61].x)
    mouth_open = abs(lm[14].y - lm[13].y)
    if mouth_width > 0.14 and mouth_open > 0.045:
        return "smile_peace"
    # Olhando para cima
    dist_nose_forehead = abs(lm[1].y - lm[10].y)
    dist_nose_chin = abs(lm[1].y - lm[152].y)
    if dist_nose_forehead < dist_nose_chin * 0.65:
        return "thinking_default"
    return None

# ---------------------------------------------------------------------------
# Detecção de gestos de mão
# ---------------------------------------------------------------------------

def detect_single_hand_gesture(hand_landmarks):
    lm = hand_landmarks.landmark
    thumb_up = lm[mp_hands.HandLandmark.THUMB_TIP].y < lm[mp_hands.HandLandmark.THUMB_IP].y
    index_up = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < lm[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    middle_up = lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    ring_up = lm[mp_hands.HandLandmark.RING_FINGER_TIP].y < lm[mp_hands.HandLandmark.RING_FINGER_PIP].y
    pinky_up = lm[mp_hands.HandLandmark.PINKY_TIP].y < lm[mp_hands.HandLandmark.PINKY_PIP].y

    if index_up and middle_up and not ring_up and not pinky_up:
        return "smile_peace"
    if thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
        return "thumbs_up"
    return None

def detect_timeout(hand_landmarks_list):
    if len(hand_landmarks_list) < 2:
        return False
    h1, h2 = hand_landmarks_list[0], hand_landmarks_list[1]
    def dist(a, b): return ((a.x - b.x)**2 + (a.y - b.y)**2)**0.5
    p1 = h1.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    p2 = h2.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    p3 = h2.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    p4 = h1.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    return dist(p1, p2) < 0.15 or dist(p3, p4) < 0.15

# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERRO: Não foi possível abrir a webcam.")
        sys.exit(1)

    memes = {k: resize_to_fit(load_meme(k, v)) for k, v in MEME_PATHS.items()}
    active_meme_key = "thinking_default"

    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh, \
         mp_hands.Hands(max_num_hands=2) as hands:
        
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok: continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = face_mesh.process(rgb)
            hand_results = hands.process(rgb)

            current_gesture = "thinking_default"

            if face_results.multi_face_landmarks:
                for face_lm in face_results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(frame, face_lm, mp_face_mesh.FACEMESH_TESSELATION,
                        mp_drawing.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=1),
                        mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    expr = detect_face_expression(face_lm)
                    if expr: current_gesture = expr

            if hand_results.multi_hand_landmarks:
                for hand_lm in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(255,105,180), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(255,105,180), thickness=2))
                    gest = detect_single_hand_gesture(hand_lm)
                    if gest: current_gesture = gest
                if detect_timeout(hand_results.multi_hand_landmarks):
                    current_gesture = "timeout"

            confirmed = stable_gesture(current_gesture)
            if confirmed: active_meme_key = confirmed

            cv2.imshow("MemeCV - Webcam", frame)
            cv2.imshow("MemeCV - Meme", memes[active_meme_key])
            if cv2.waitKey(5) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
