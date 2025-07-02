import tkinter as tk
from tkinter import messagebox
import cv2
import face_recognition
import numpy as np
import json
import os
import random

DISTANCIA_INTERPUPILAR_MM = 63  # média interpupilar em adultos

def euclidean_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def capturar_e_processar_foto(nome_formatado):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Erro", "Não foi possível acessar a câmera.")
        return None, None

    messagebox.showinfo("Captura", "Pressione ESPAÇO para capturar a imagem.")
    foto = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Captura - Pressione ESPAÇO", frame)
        key = cv2.waitKey(1)
        if key == 32:  # Espaço
            foto = frame.copy()
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if foto is None:
        return None, None

    rgb_frame = foto[:, :, ::-1]
    face_landmarks_list = face_recognition.face_landmarks(rgb_frame)
    if not face_landmarks_list:
        messagebox.showerror("Erro", "Nenhum rosto detectado.")
        return None, None

    face_landmarks = face_landmarks_list[0]
    left_eye = face_landmarks.get('left_eye')
    right_eye = face_landmarks.get('right_eye')

    if left_eye is None or right_eye is None:
        messagebox.showerror("Erro", "Não foi possível identificar os olhos.")
        return None, None

    # Medidas
    left_eye_outer = left_eye[0]
    right_eye_outer = right_eye[-1]
    largura_armacao_px = euclidean_dist(left_eye_outer, right_eye_outer)
    left_eye_top = min(left_eye, key=lambda p: p[1])
    left_eye_bottom = max(left_eye, key=lambda p: p[1])
    altura_lente_px = euclidean_dist(left_eye_top, left_eye_bottom)

    # Maior diagonal
    max_diag = 0
    diag_pts = (left_eye[0], left_eye[3])
    for i in range(len(left_eye)):
        for j in range(i+1, len(left_eye)):
            d = euclidean_dist(left_eye[i], left_eye[j])
            if d > max_diag:
                max_diag = d
                diag_pts = (left_eye[i], left_eye[j])
    maior_diagonal_px = max_diag

    # Distância pupilar
    left_eye_center = np.mean(left_eye, axis=0)
    right_eye_center = np.mean(right_eye, axis=0)
    distancia_olhos_px = euclidean_dist(left_eye_center, right_eye_center)

    px_to_mm = DISTANCIA_INTERPUPILAR_MM / distancia_olhos_px
    largura_armacao_mm = largura_armacao_px * px_to_mm
    altura_lente_mm = altura_lente_px * px_to_mm
    maior_diagonal_mm = maior_diagonal_px * px_to_mm
    distancia_olhos_mm = distancia_olhos_px * px_to_mm

    X = largura_armacao_mm - distancia_olhos_mm
    Y = X + maior_diagonal_mm
    Z = Y + 32
    altura_montagem_mm = abs((left_eye_center[1] - left_eye_bottom[1]) * px_to_mm)

    # Desenhar
    cv2.line(foto, tuple(left_eye_outer), tuple(right_eye_outer), (255, 0, 0), 2)
    cv2.line(foto, tuple(left_eye_top), tuple(left_eye_bottom), (0, 0, 255), 2)
    cv2.line(foto, tuple(np.int32(diag_pts[0])), tuple(np.int32(diag_pts[1])), (0, 255, 0), 2)
    cv2.line(foto, tuple(left_eye_center.astype(int)), tuple(right_eye_center.astype(int)), (0, 255, 255), 2)

    # Textos
    cv2.putText(foto, f'P+A: {largura_armacao_mm:.1f}mm', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    cv2.putText(foto, f'H: {altura_lente_mm:.1f}mm', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.putText(foto, f'MD: {maior_diagonal_mm:.1f}mm', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(foto, f'DP: {distancia_olhos_mm:.1f}mm', (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(foto, f'X: {X:.1f}mm', (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,128,0), 2)
    cv2.putText(foto, f'Y: {Y:.1f}mm', (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128,0,255), 2)
    cv2.putText(foto, f'Z: {Z:.1f}mm', (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,128), 2)
    cv2.putText(foto, f'Alt. montagem: {altura_montagem_mm:.1f}mm', (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

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

    # Salvar imagem com nome
    img_path = f"{nome_formatado}.jpg"
    cv2.imwrite(img_path, foto)

    return medidas, img_path

def salvar_tudo():
    nome = entry_nome.get().strip()
    nome_formatado = nome.replace(" ", "_")
    # Gera número aleatório de 4 dígitos
    numero_aleatorio = random.randint(1000, 9999)
    # Cria pasta do cadastro
    pasta_base = os.path.join("cadastros", f"{nome_formatado}_{numero_aleatorio}")
    os.makedirs(pasta_base, exist_ok=True)
    miopia_esq = entry_miopia_esq.get().strip()
    miopia_dir = entry_miopia_dir.get().strip()
    astig_esq = entry_astig_esq.get().strip()
    astig_dir = entry_astig_dir.get().strip()

    if not nome:
        messagebox.showerror("Erro", "O campo nome é obrigatório.")
        return

    try:
        float(miopia_esq)
        float(miopia_dir)
        float(astig_esq)
        float(astig_dir)
    except ValueError:
        messagebox.showerror("Erro", "Os campos de grau devem conter números.")
        return

    medidas, img_path = capturar_e_processar_foto(nome_formatado)
    if medidas is None:
        return

    # Salvar em .txt dentro da pasta do cadastro
    txt_path = os.path.join(pasta_base, "cadastro.txt")
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(
            f"Nome: {nome}\n"
            f"Miopia - OE: {miopia_esq} | OD: {miopia_dir}\n"
            f"Astigmatismo - OE: {astig_esq} | OD: {astig_dir}\n"
            f"Imagem: {img_path}\n"
            "------------------------------------------\n"
        )

    # Arredondar medidas para 2 casas decimais
    medidas_2casas = {k: round(v, 2) for k, v in medidas.items()}

    # Salvar JSON dentro da pasta do cadastro
    json_path = os.path.join(pasta_base, f"{nome_formatado}_medidas.json")
    with open(json_path, "w") as f:
        json.dump(medidas_2casas, f, indent=4)

    # Mover imagem para a pasta do cadastro
    if img_path and os.path.exists(img_path):
        new_img_path = os.path.join(pasta_base, os.path.basename(img_path))
        os.replace(img_path, new_img_path)

    messagebox.showinfo("Sucesso", f"Cadastro completo salvo em {pasta_base}!")
    limpar_campos()

def limpar_campos():
    entry_nome.delete(0, tk.END)
    entry_miopia_esq.delete(0, tk.END)
    entry_miopia_dir.delete(0, tk.END)
    entry_astig_esq.delete(0, tk.END)
    entry_astig_dir.delete(0, tk.END)

# Janela UI
janela = tk.Tk()
janela.title("Cadastro Óculos com Medições Faciais")

tk.Label(janela, text="Nome:").grid(row=0, column=0)
entry_nome = tk.Entry(janela, width=30)
entry_nome.grid(row=0, column=1, columnspan=3)

tk.Label(janela, text="Miopia OE:").grid(row=1, column=0)
entry_miopia_esq = tk.Entry(janela)
entry_miopia_esq.grid(row=1, column=1)

tk.Label(janela, text="Miopia OD:").grid(row=1, column=2)
entry_miopia_dir = tk.Entry(janela)
entry_miopia_dir.grid(row=1, column=3)

tk.Label(janela, text="Astigmatismo OE:").grid(row=2, column=0)
entry_astig_esq = tk.Entry(janela)
entry_astig_esq.grid(row=2, column=1)

tk.Label(janela, text="Astigmatismo OD:").grid(row=2, column=2)
entry_astig_dir = tk.Entry(janela)
entry_astig_dir.grid(row=2, column=3)

btn_salvar = tk.Button(janela, text="Capturar & Salvar", command=salvar_tudo, bg="lightgreen")
btn_salvar.grid(row=3, column=1, pady=10)

btn_limpar = tk.Button(janela, text="Limpar Campos", command=limpar_campos)
btn_limpar.grid(row=3, column=2, pady=10)

janela.mainloop()
