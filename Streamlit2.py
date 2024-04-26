import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image 
from time import sleep
import os 
import base64
from clickhouse_driver import Client
from deepface import DeepFace
from datetime import datetime


def daftar():
    st.title("**Face Registration**")
    b2_disabled = True  # Set default value for b2_disabled
    def empty():
        ph.empty()
        with st.spinner('Loading...'):
            sleep(0.1)

    app_mode =  st.sidebar.selectbox("choose the app mode",
                                        ["Registration"])

    if app_mode == "Registration":

        b1 = False
        b2 = False
        b3 = False
        b4 = False
        ph = st.empty()

    if "b1" not in st.session_state:
        st.session_state.b1 = False 
    if "b2" not in st.session_state:
        st.session_state.b2 = False
    if "b3" not in st.session_state:
        st.session_state.b3 = False
    if "b4" not in st.session_state:
        st.session_state.b4 = False

    with ph.container():
        b1 = st.button("Mulai Registrasi", on_click=Mulai_registrasi)

    if st.session_state.b1 or b1:
        empty()
        with ph.container():
            st.header("Step 2")
            with st.form(key='my_form'):
                st.subheader("Tolong Masukan Nama dan Id anda")

                # Mendefinisikan input nama
                nama = st.text_input("Nama")

                # Mendefinisikan input ID
                id = st.text_input("ID")

                # Submit button
                submit_button = st.form_submit_button(label='Check')
                if submit_button:
                    if not nama.strip() or not id.strip():
                        st.error("Mohon isi nama dan ID.")
                    else:
                        with st.spinner('Loading...'):
                            sleep(1)
                        st.success("Input berhasil disubmit!")
                        st.write("Nama:", nama)
                        st.write("ID:", id)
                        st.session_state.nama = nama
                        st.session_state.id = id
                        b2_disabled = False  # Inisialisasi nilai tombol Next
                else:
                    b2_disabled = True  # Menonaktifkan tombol Next jika formulir belum disubmit

            # Tombol Next dengan status dinamis berdasarkan apakah formulir telah disubmit dan data terpenuhi
            b2 = st.button("Next", on_click=capture_and_detect_faces, disabled=b2_disabled)

    if st.session_state.b2 or b2:
        empty()
        with ph.container():
            b3 = st.button("Show Photo?", on_click=checkphoto)

    if st.session_state.b3 or b3:
        empty()
        with ph.container():
            b4, col2 = st.columns(2)
            # Button to go back to step 1
            if b4.button("Insert to Database", key="back_to_step_1_button", on_click=insert_to_database):
                pass  
            if col2.button("Take image again", key="take_image_again_button", on_click=reset_step_2_and_detect_faces):
                pass  

    if st.session_state.b4 or b4:
        empty()
        with ph.container():
            display_registration_message()


@st.cache_data()
def image_resize(image, width = None, height = None, inter =  cv2.INTER_AREA):
        dim = None
        (h,w) = image.shape[:2]

        if width is None and height is None:
            return image
        
        if width is None:
            r= width/float(w)
            dim = (int(w*r), height)

        else:
            r = width/float(w)
            dim = (width, int(w*r))

        #rezise the image
        resized = cv2.resize(image, dim, interpolation=inter)

        return resized

# Fungsi untuk membuat folder temporary_photo
def create_temporary_folder():
    folder_path = "./temporary_photo"
    
    # Cek apakah folder temporary_photo sudah ada
    if os.path.exists(folder_path):
        # Jika sudah ada, hapus semua file di dalamnya
        [os.remove(os.path.join(folder_path, file)) for file in os.listdir(folder_path)]
    else:
        # Jika belum ada, buat folder temporary_photo
        os.makedirs(folder_path)

# Fungsi untuk membuat folder transformed_images
def create_transformed_folder():
    folder_path = "transformed_images"
    
    # Cek apakah folder transformed_images sudah ada
    if os.path.exists(folder_path):
        # Jika sudah ada, hapus semua file di dalamnya
        [os.remove(os.path.join(folder_path, file)) for file in os.listdir(folder_path)]
    else:
        # Jika belum ada, buat folder transformed_images
        os.makedirs(folder_path)

# Fungsi untuk menerapkan transformasi ke gambar-gambar dalam folder temporary_photo
def apply_transformations_to_temporary_images(temporary_photo_dir):
    # Tentukan direktori output
    output_dir = "transformed_images"
    os.makedirs(output_dir, exist_ok=True)

    # Definisikan transformasi yang akan dilakukan
    transformations = [
        ("brighten", lambda img: cv2.convertScaleAbs(img, beta=50)),
        ("darken", lambda img: cv2.convertScaleAbs(img, beta=-50)),
        ("high_contrast", lambda img: cv2.convertScaleAbs(img, alpha=1.5, beta=0)),
        ("low_contrast", lambda img: cv2.convertScaleAbs(img, alpha=0.6, beta=0)),
        ("blur", lambda img: cv2.GaussianBlur(img, (15, 15), 0)),
        ("sharpen", lambda img: cv2.filter2D(img, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))),
        ("add_noise", lambda img: cv2.add(img, np.random.normal(0, 0.5, img.shape).astype(np.uint8))),
        ("rotate_only", lambda img: img)
    ]


    # Tentukan sudut rotasi
    rotation_angles = [0, 45, 315]

    # Ambil daftar file gambar di dalam direktori temporary_photo
    image_files = [filename for filename in os.listdir(temporary_photo_dir) if filename.endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    # Terapkan transformasi ke setiap gambar di dalam direktori temporary_photo
    for image_file in image_files:
        image_path = os.path.join(temporary_photo_dir, image_file)
        image = cv2.imread(image_path)

        # Iterasi melalui setiap transformasi dan sudut rotasi
        for name, transform_func in transformations:
            for angle in rotation_angles:
                # Putar gambar
                rotated_image = cv2.warpAffine(image, cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1), (image.shape[1], image.shape[0]))

                # Terapkan transformasi
                transformed_image = transform_func(rotated_image.copy())

                # Simpan gambar yang telah ditransformasi
                output_path = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_rotated_{angle}_{name}.jpg")
                cv2.imwrite(output_path, transformed_image)


def check_quality(frame, tresshold_blur, tresshold_dark, tresshold_bright, kpi3_text, kpi4_text):
    # Inisialisasi threshold untuk deteksi blur, kegelapan, dan kecerahan
    # Konversi gambar ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Hitung nilai laplacian dan rata-rata pixel gambar
    laplacian = np.var(cv2.Laplacian(gray, cv2.CV_64F))
    average_pixel_value = np.mean(gray)
    if tresshold_blur < laplacian:
            kpi4_text.write(f"<h1 style='text-align: center; color:green;'>{int(laplacian)}</h1>", unsafe_allow_html=True)
    if tresshold_blur > laplacian:
            kpi4_text.write(f"<h1 style='text-align: center; color:red;'>{int(laplacian)}!</h1>", unsafe_allow_html=True)
    if average_pixel_value < tresshold_bright and tresshold_dark < average_pixel_value:
            kpi3_text.write(f"<h1 style='text-align: center; color:green;'>{int(average_pixel_value)}</h1>", unsafe_allow_html=True)
    if average_pixel_value > tresshold_bright:
            kpi3_text.write(f"<h1 style='text-align: center; color:red;'>{int(average_pixel_value)}! Terlalu Terang</h1>", unsafe_allow_html=True)
    if tresshold_dark > average_pixel_value:
            kpi3_text.write(f"<h1 style='text-align: center; color:red;'>{int(average_pixel_value)}! Terlalu Gelap</h1>", unsafe_allow_html=True)
    if average_pixel_value < tresshold_bright and tresshold_dark < average_pixel_value and tresshold_blur < laplacian: 
            kpi3_text.write(f"<h1 style='text-align: center; color:green;'>{int(average_pixel_value)}</h1>", unsafe_allow_html=True)
            kpi4_text.write(f"<h1 style='text-align: center; color:green;'>{int(laplacian)}</h1>", unsafe_allow_html=True)

    return average_pixel_value < tresshold_bright and tresshold_dark < average_pixel_value and tresshold_blur < laplacian

def show_images_in_folder(folder_path):
    # Get a list of image files in the folder
    image_files = [filename for filename in os.listdir(folder_path) if filename.endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    # Display each image
    for filename in image_files:
        st.image(os.path.join(folder_path, filename))


# Fungsi untuk mengubah gambar menjadi vektor representasi dan base64
def image_to_vector(img_path):
    try:
        # Gunakan DeepFace untuk mendapatkan vektor representasi dan base64
        result = DeepFace.represent(img_path, 
                                    model_name = "Facenet512",
                                    enforce_detection = False,
                                    detector_backend = "fastmtcnn")
        
        # Ambil nilai dari kunci pertama
        vector_representation = result[0]['embedding']

        # Ubah img_path ke base64
        with open(img_path, "rb") as image_file:
            base64_encoded = base64.b64encode(image_file.read()).decode('utf-8')

        return vector_representation, base64_encoded
    except Exception as e:
        print("Error:1", str(e))
        return None, None
    pass

# Fungsi untuk menyimpan data karyawan ke tabel ClickHouse
def save_employee_data(employee_id, employee_name, vector_representation, base64_encoded):
    try:
        # Inisialisasi koneksi ClickHouse
        client = Client(host='localhost', port=9000, user='default', database='default')
        
        # Query untuk menyimpan data ke tabel karyawan
        query = f"INSERT INTO employee (employee_id, employee_name, embedding, image_base64) VALUES ({employee_id}, '{employee_name}', {vector_representation}, '{base64_encoded}');"

        # Eksekusi query
        client.execute(query)
    except Exception as e:
        print(f"Error:2 {e}")
    pass

# Fungsi untuk memasukkan data karyawan ke database
def insert_ke_database():
    # Direktori tempat Anda menyimpan 20 foto
    input_directory = 'transformed_images'
    employee_name = st.session_state.nama
    employee_id = st.session_state.id
    # Mengecek apakah direktori ada dan tidak kosong
    if os.path.exists(input_directory) and os.listdir(input_directory):
        # Iterasi setiap file dalam direktori
        for filename in os.listdir(input_directory):
            if filename.endswith(".jpg"):
                # Path lengkap ke file
                img_path = os.path.join(input_directory, filename)

                # Menggunakan fungsi image_to_vector untuk mendapatkan vektor dan base64
                vector_representation, base64_encoded = image_to_vector(img_path)

                # Menggunakan fungsi save_employee_data untuk menyimpan data ke ClickHouse
                if vector_representation is not None and base64_encoded is not None:
                    save_employee_data(employee_id, employee_name, vector_representation, base64_encoded)
    else:
       pass

# Fungsi untuk menangkap wajah dari kamera dan menyimpannya
def Mulai_registrasi():
    create_temporary_folder()
    st.session_state.b1 = True

def capture_and_detect_faces():
    create_temporary_folder()
    st.session_state.b2 = True

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    tracking_confidence = 0.5
    detection_confidence = 0.5
    tresshold_blur = 20
    tresshold_bright = 150
    tresshold_dark = 0

    stframe = st.empty()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.markdown("**Frame Rate**")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("**Detected Face**")
        kpi2_text = st.markdown("0")

    with kpi3:
        st.markdown("**Image Brightness**")
        kpi3_text = st.markdown("0")

    with kpi4:
        st.markdown("**Image Resolution**")
        kpi4_text = st.markdown("0")

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence
    ) as face_mesh:
        prevTime = 0
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                continue

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            results = face_mesh.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            img_h, img_w, _ = frame.shape

            face_count = 0
            face_3d = []
            face_2d = []
            fps = 0

            if results.multi_face_landmarks is not None and len(results.multi_face_landmarks) > 0:
                if check_quality(frame, tresshold_blur, tresshold_dark, tresshold_bright, kpi3_text, kpi4_text):
                    for face_landmarks in results.multi_face_landmarks:
                        face_count += 1
                        for idx, lm in enumerate(face_landmarks.landmark):
                            if idx in [1, 33, 263, 62, 291, 199]:
                                x, y = int(lm.x * img_w), int(lm.y * img_h)
                                face_2d.append([x, y])
                                face_3d.append([x, y, lm.z])
                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)

                    focal_length = 1 * img_w
                    cam_matrix = np.array([[focal_length, 0, img_w/2],
                                           [0, focal_length, img_h/2],
                                           [0, 0, 1]])
                    dist_coeffs = np.zeros((4, 1))
                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_coeffs)
                    rmat, _ = cv2.Rodrigues(rot_vec)
                    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                    x, y, z = angles[0] * 360, angles[1] * 360, angles[2] * 360

                    if x < 10 and x > -10 and y < 10 and y > -10 and z < 5 and z > -5: 
                        text = "looking_forward"
                        # Ambil gambar jika kondisi terpenuhi dan gambar belum ada
                        save_path = r".\temporary_photo\looking_forward.jpg"
                        if not os.path.exists(save_path):
                            cv2.imwrite(save_path, frame)

                    elif x < 5 and x > -5 and y < -10 and y > -25 and z < 5 and z > -5: 
                        text = "looking_tilty_left"
                        # Ambil gambar jika kondisi terpenuhi dan gambar belum ada
                        save_path = r"./temporary_photo/looking_tilty_left.jpg"
                        if not os.path.exists(save_path):
                            cv2.imwrite(save_path, frame)

                    elif x < 5 and x > -5  and y < -25 and z < 5 and z > -5: 
                        text = "looking_left"
                        # Ambil gambar jika kondisi terpenuhi dan gambar belum ada
                        save_path = r"./temporary_photo/looking_left.jpg"
                        if not os.path.exists(save_path):
                            cv2.imwrite(save_path, frame)

                    elif x < 5 and x > -5 and y > 10 and y < 25 and z < 5 and z > -5: 
                        text = "looking_tilty_right"
                        # Ambil gambar jika kondisi terpenuhi dan gambar belum ada
                        save_path = r"./temporary_photo/looking_tilty_right.jpg"
                        if not os.path.exists(save_path):
                            cv2.imwrite(save_path, frame)

                    elif x < 5 and x > -5  and y > 25 and z < 5 and z > -5: 
                        text = "looking_right"
                        # Ambil gambar jika kondisi terpenuhi dan gambar belum ada
                        save_path = r"./temporary_photo/looking_right.jpg"
                        if not os.path.exists(save_path):
                            cv2.imwrite(save_path, frame)

                    elif x < -10 and y < 8 and y > -8: 
                        text = "looking_down"
                        # Ambil gambar jika kondisi terpenuhi dan gambar belum ada
                        save_path = r".\temporary_photo\looking_down.jpg"
                        if not os.path.exists(save_path):
                            cv2.imwrite(save_path, frame)

                    elif x < -5 and y < -8 and z < 5 and z > -5: 
                        text = "looking_down_tilty_left"
                        # Ambil gambar jika kondisi terpenuhi dan gambar belum ada
                        save_path = r".\temporary_photo\looking_down_tilty_left.jpg"
                        if not os.path.exists(save_path):
                            cv2.imwrite(save_path, frame)

                    elif x < -5 and y > 8 and z < 5 and z > -5: 
                        text = "looking_down_tilty_right"
                        # Ambil gambar jika kondisi terpenuhi dan gambar belum ada
                        save_path = r".\temporary_photo\looking_down_tilty_right.jpg"
                        if not os.path.exists(save_path):
                            cv2.imwrite(save_path, frame)

                currTime = time.time()
                fps = 1 / (currTime - prevTime)
                prevTime = currTime

                kpi1_text.write(f"<h1 style='text-align: center; color:green;'>{int(fps)}</h1>", unsafe_allow_html=True)
                kpi2_text.write(f"<h1 style='text-align: center; color:green;'>{face_count}</h1>", unsafe_allow_html=True)
                check_quality(frame, tresshold_blur, tresshold_dark, tresshold_bright, kpi3_text, kpi4_text)

            frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
            stframe.image(frame, channels='BGR', use_column_width=True)

            folder_path = "temporary_photo"
            image_files = [(filename, cv2.imread(os.path.join(folder_path, filename))) for filename in os.listdir(folder_path) if filename.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
            if len(image_files) == 8:
                return True

    return False           

def reset_step_2_and_detect_faces():
    st.session_state.b2 = False
    st.session_state.b3 = False
    st.session_state.b4 = False
    capture_and_detect_faces()

def checkphoto():
    st.session_state.b3 = True
    # Usage example
    folder_path = "temporary_photo"
    show_images_in_folder(folder_path)
    

def insert_to_database():
    create_transformed_folder()
    apply_transformations_to_temporary_images("temporary_photo")
    insert_ke_database()
    st.session_state.b4 = True
    

def display_registration_message():
    st.success('Your face hase been registered', icon="✅")

if __name__ == "__main__":
    daftar()

