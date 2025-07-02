import cv2
import numpy as np
import face_recognition
from utils import euclidean_dist

DISTANCIA_INTERPUPILAR_MM = 63  # valor médio para adultos

def detectar_face_landmarks(foto):
    rgb_frame = foto[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_landmarks_list = face_recognition.face_landmarks(rgb_frame)
    return face_locations, face_landmarks_list

def calcular_medidas(foto, face_landmarks):
    left_eye = face_landmarks.get('left_eye')
    right_eye = face_landmarks.get('right_eye')
    nose_bridge = face_landmarks.get('nose_bridge')

    if left_eye is None or right_eye is None:
        return None, 'Óculos não detectado'

    # Largura da armação (P+A): distância entre o canto externo dos olhos
    left_eye_outer = left_eye[0]
    right_eye_outer = right_eye[-1]
    largura_armacao_px = euclidean_dist(left_eye_outer, right_eye_outer)
    cv2.line(foto, tuple(left_eye_outer), tuple(right_eye_outer), (255, 0, 0), 2)
    pos_p_a = (min(left_eye_outer[0], right_eye_outer[0]) + abs(left_eye_outer[0] - right_eye_outer[0]) // 2, min(left_eye_outer[1], right_eye_outer[1]) - 10)
    cv2.putText(foto, 'P+A', pos_p_a, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    # Altura da lente (H): maior distância vertical dentro do olho esquerdo
    left_eye_top = min(left_eye, key=lambda p: p[1])
    left_eye_bottom = max(left_eye, key=lambda p: p[1])
    altura_lente_px = euclidean_dist(left_eye_top, left_eye_bottom)
    pos_h = (left_eye_top[0] - 40, (left_eye_top[1] + left_eye_bottom[1]) // 2)
    cv2.line(foto, tuple(left_eye_top), tuple(left_eye_bottom), (0, 0, 255), 2)
    cv2.putText(foto, 'H', pos_h, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # Maior diagonal (MD) dentro do olho esquerdo
    max_diag = 0
    diag_pts = (left_eye[0], left_eye[3])
    for i in range(len(left_eye)):
        for j in range(i+1, len(left_eye)):
            d = euclidean_dist(left_eye[i], left_eye[j])
            if d > max_diag:
                max_diag = d
                diag_pts = (left_eye[i], left_eye[j])
    maior_diagonal_px = max_diag
    pos_md = ((diag_pts[0][0] + diag_pts[1][0]) // 2 + 10, (diag_pts[0][1] + diag_pts[1][1]) // 2 - 10)
    cv2.line(foto, tuple(np.int32(diag_pts[0])), tuple(np.int32(diag_pts[1])), (0, 255, 0), 2)
    cv2.putText(foto, 'MD', pos_md, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Distância pupilar (DP): centro a centro dos olhos
    left_eye_center = np.mean(left_eye, axis=0)
    right_eye_center = np.mean(right_eye, axis=0)
    distancia_olhos_px = euclidean_dist(left_eye_center, right_eye_center)
    pos_dp = (int((left_eye_center[0] + right_eye_center[0]) // 2), int((left_eye_center[1] + right_eye_center[1]) // 2) - 10)
    cv2.line(foto, tuple(left_eye_center.astype(int)), tuple(right_eye_center.astype(int)), (0, 255, 255), 2)
    cv2.putText(foto, 'DP', pos_dp, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    # Conversão de pixels para mm
    px_to_mm = DISTANCIA_INTERPUPILAR_MM / distancia_olhos_px
    largura_armacao_mm = largura_armacao_px * px_to_mm
    altura_lente_mm = altura_lente_px * px_to_mm
    maior_diagonal_mm = maior_diagonal_px * px_to_mm
    distancia_olhos_mm = distancia_olhos_px * px_to_mm

    X = largura_armacao_mm - distancia_olhos_mm
    Y = X + maior_diagonal_mm
    Z = Y + 32
    altura_montagem_mm = abs((left_eye_center[1] - left_eye_bottom[1]) * px_to_mm)

    medidas = {
        'largura_armacao_mm': largura_armacao_mm,
        'altura_lente_mm': altura_lente_mm,
        'maior_diagonal_mm': maior_diagonal_mm,
        'distancia_pupilar_mm': distancia_olhos_mm,
        'X_mm': X,
        'Y_mm': Y,
        'Z_mm': Z,
        'altura_montagem_mm': altura_montagem_mm
    }
    return medidas, None

def desenhar_medidas(foto, medidas):
    # Barra lateral translúcida
    overlay = foto.copy()
    bar_width = 260
    alpha = 0.55
    cv2.rectangle(overlay, (0, 0), (bar_width, foto.shape[0]), (30, 30, 30), -1)
    cv2.addWeighted(overlay, alpha, foto, 1 - alpha, 0, foto)

    # Parâmetros de layout
    x0 = 20
    y0 = 40
    dy = 38
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.95
    thickness = 2

    # Medidas formatadas
    textos = [
        (f"P+A: {medidas['largura_armacao_mm']:.1f}mm", (255, 153, 51)),
        (f"H: {medidas['altura_lente_mm']:.1f}mm", (255, 51, 51)),
        (f"MD: {medidas['maior_diagonal_mm']:.1f}mm", (51, 255, 102)),
        (f"DP: {medidas['distancia_pupilar_mm']:.1f}mm", (255, 255, 51)),
        (f"X: {medidas['X_mm']:.1f}mm", (102, 178, 255)),
        (f"Y: {medidas['Y_mm']:.1f}mm", (255, 102, 255)),
        (f"Z: {medidas['Z_mm']:.1f}mm", (255, 102, 178)),
        (f"Alt. montagem: {medidas['altura_montagem_mm']:.1f}mm", (255, 255, 255)),
    ]
    for i, (texto, cor) in enumerate(textos):
        y = y0 + i * dy
        # Sombra preta para contraste
        cv2.putText(foto, texto, (x0+2, y+2), font, font_scale, (0,0,0), thickness+2, cv2.LINE_AA)
        cv2.putText(foto, texto, (x0, y), font, font_scale, cor, thickness, cv2.LINE_AA)