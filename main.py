import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
# Standart ttk yerine ttkbootstrap'ı 'ttk' olarak içe aktarıyoruz
import ttkbootstrap as ttk
# Buton renkleri gibi sabitleri içe aktarıyoruz
from ttkbootstrap.constants import *
from PIL import Image, ImageTk

# --- Backend: MediaPipe modülleri (Burası değişmedi) ---
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class HandFaceApp:
    def __init__(self, root_window):
        # root_window zaten bir ttk.Window, bu yüzden 'self.root_window' olarak atıyoruz
        self.root_window = root_window
        self.root_window.title("Akıllı Sınıf Takip Sistemi v3.0 (Bootstrap)")
        self.root_window.geometry("1300x820") # Boyutu temaya göre hafifçe ayarladık
        
        # --- Tasarım: 'setup_styles' fonksiyonuna GEREK KALMADI ---
        # Tema ('darkly') tüm stilleri otomatik hallediyor.
        
        self.create_widgets()

        # Backend değişkenleri (Burası değişmedi)
        self.cap = None
        self.hands_model = None
        self.face_mesh_model = None
        self.is_camera_running = False
        
        # Güvenli kapatma (Burası değişmedi)
        self.root_window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        """Arayüz elemanlarını oluşturur ve yerleştirir."""
        
        # 1. Başlık Alanı
        # padding (py) ve font büyüklüğü (bootstyle) bootstrap ile ayarlandı
        self.title_label = ttk.Label(self.root_window, 
                                     text="Akıllı Sınıf Takip Sistemi", 
                                     bootstyle="primary", # Temanın ana rengini kullan
                                     font=("Helvetica", 28, "bold"))
        self.title_label.pack(pady=(20, 10)) # Üstten 20, alttan 10 boşluk

        # 2. Video Görüntüsü Alanı
        # Çerçeveye 'primary' renginde bir kenarlık ekledik
        self.video_frame = ttk.Frame(self.root_window, 
                                     width=1280, 
                                     height=720,
                                     bootstyle="primary", # Kenarlık rengi
                                     padding=2)
        self.video_frame.pack(pady=10, padx=10)
        self.video_frame.pack_propagate(False) 
        
        # Görüntü alanı (Label), arkaplanı 'dark' olacak
        self.video_label = ttk.Label(self.video_frame, bootstyle="dark")
        self.video_label.pack(expand=True, fill='both') 

        # 3. Kontrol Butonları Alanı
        self.button_frame = ttk.Frame(self.root_window)
        self.button_frame.pack(pady=20)

        # Başlat Butonu
        self.start_button = ttk.Button(self.button_frame, 
                                       text="Kamerayı Başlat", 
                                       bootstyle="success", # OTOMATİK YEŞİL
                                       command=self.start_camera)
        self.start_button.pack(side=tk.LEFT, padx=20)

        # Durdur Butonu (Kapatma tuşu)
        self.stop_button = ttk.Button(self.button_frame, 
                                      text="Kamerayı Durdur", 
                                      bootstyle="danger", # OTOMATİK KIRMIZI
                                      command=self.stop_camera,
                                      state=DISABLED) # Başlangıçta pasif
        self.stop_button.pack(side=tk.LEFT, padx=20)

    # --- Backend Fonksiyonları (Hiçbir değişiklik yapılmadı) ---

    def start_camera(self):
        if self.is_camera_running:
            return 
        print("Kamera başlatılıyor...")
        self.is_camera_running = True
        
        self.start_button.config(state=DISABLED)
        self.stop_button.config(state=NORMAL)
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.hands_model = mp_hands.Hands(
            static_image_mode=False, max_num_hands=10, 
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
            
        self.face_mesh_model = mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=10, 
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
            
        self.update_frame()

    def stop_camera(self):
        if not self.is_camera_running:
            return
        print("Kamera durduruluyor...")
        self.is_camera_running = False
        
        self.start_button.config(state=NORMAL)
        self.stop_button.config(state=DISABLED)

        if self.cap:
            self.cap.release()
        if self.hands_model:
            self.hands_model.close()
        if self.face_mesh_model:
            self.face_mesh_model.close()
            
        self.video_label.config(image='')
        # Görüntü yokken arkaplanı 'dark' yap (temadan)
        self.video_label.configure(bootstyle="dark") 

    def on_closing(self):
        print("Pencere kapatılıyor...")
        self.stop_camera() 
        self.root_window.destroy() 

    def update_frame(self):
        if not self.is_camera_running:
            return

        success, image = self.cap.read()
        if not success:
            print("Kamera karesi okunamıyor.")
            self.root_window.after(10, self.update_frame)
            return

        # --- MEDIAPIPE BACKEND (Aynen kopyalandı) ---
        image_rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        hand_results = self.hands_model.process(image_rgb)
        face_mesh_results = self.face_mesh_model.process(image_rgb)
        
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        image_bgr.flags.writeable = True

        toplam_el_sayisi = 0
        toplam_yuz_sayisi = 0

        if face_mesh_results.multi_face_landmarks:
            toplam_yuz_sayisi = len(face_mesh_results.multi_face_landmarks)
            for face_landmarks in face_mesh_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image_bgr, landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

        if hand_results.multi_hand_landmarks:
            toplam_el_sayisi = len(hand_results.multi_hand_landmarks)
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=image_bgr, landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())

        cv2.putText(image_bgr, f'TESPIT EDILEN YUZ: {toplam_yuz_sayisi}', 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image_bgr, f'TESPIT EDILEN EL: {toplam_el_sayisi}', 
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        
        # --- MEDIAPIPE BACKEND BİTİŞİ ---

        cv2_image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv2_image_rgb)
        imgtk = ImageTk.PhotoImage(image=pil_image)

        self.video_label.imgtk = imgtk 
        self.video_label.configure(image=imgtk)

        self.root_window.after(10, self.update_frame)

# Ana programı başlat
if __name__ == "__main__":
    # 1. Ana pencereyi 'ttk.Window' olarak oluşturuyor ve temayı ('darkly') seçiyoruz
    # Diğer tema seçenekleri: 'litera', 'minty', 'superhero', 'cyborg'
    root = ttk.Window(themename="darkly") 
    
    # 2. Uygulamamızı bu pencereye bağlıyoruz
    app = HandFaceApp(root) 
    
    # 3. Uygulamayı çalıştırıyoruz
    root.mainloop()