import os
import time
import cv2
import base64
import numpy as np
from clickhouse_driver import Client
from deepface import DeepFace
from datetime import datetime

def get_face_representation(frame):
    # Save the frame as an image file
    image_path = 'temp_frame.jpg'
    cv2.imwrite(image_path, frame)

    # detected_face = DeepFace.detectFace(image_path=image_path, detector_backend='opencv')
    representation = DeepFace.represent(img_path=image_path, model_name='Facenet512', enforce_detection=False, detector_backend='fastmtcnn')
    face_representation = representation[0]['embedding']
    facial_area = list(representation[0]['facial_area'].values())
    base64_representation = base64.b64encode(cv2.imencode('.jpg', cv2.imread(image_path))[1]).decode()

    # Remove the temporary image file
    os.remove(image_path)

    return face_representation, base64_representation, facial_area

def record_attendance(employee_id, employee_name):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    client.execute(
        f'INSERT INTO kehadiran (employee_id, employee_name, waktu_kehadiran) VALUES ({employee_id}, \'{employee_name}\', \'{current_time}\')'
    )

def update_or_insert_employee(employee_id, employee_name, embedding, base64_representation):
    client.execute(
        f'INSERT INTO employee (employee_id, employee_name, embedding, image_base64) '
        f'VALUES ({employee_id}, \'{employee_name}\', {embedding}, \'{base64_representation}\') '
    )

def display_attendance_info(frame, employee_id, employee_name, cosine_similarity, facial_area):
    # Gambar bounding box untuk wajah
    x, y, w, h = facial_area[0], facial_area[1], facial_area[2], facial_area[3]
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Tambahkan teks nama di bawah bounding box
    cv2.putText(frame, employee_name, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Tambahkan teks similarity di bawah nama
    cv2.putText(frame, f'Similarity: {format(cosine_similarity, ".2%")}', (x, y + h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Tampilkan frame di jendela
    #cv2.imshow('Capture Face', frame)
    return frame

def cek_kehadiran(employee_id, employee_name):
    # Ambil tanggal hari ini
    today_date = datetime.now().strftime('%Y-%m-%d')

    # Query SQL untuk memeriksa kehadiran
    query = f"SELECT * FROM kehadiran WHERE employee_id = '{employee_id}' AND employee_name = '{employee_name}' AND waktu_kehadiran >= '{today_date}'"

    # Eksekusi query dengan menggunakan variabel client
    hadir = client.execute(query)

    return hadir # if len(hadir) > 0 maka sudah hadir

def compare_faces(frame, embedding, facial_area):
    query_result = client.query_dataframe(
        f'SELECT employee_id, employee_name, 1 - cosineDistance({embedding}, embedding) AS cosineSimilarity, image_base64 FROM employee ORDER BY cosineSimilarity DESC LIMIT 1'
    )

    if not query_result.empty:
        result_employee_id, result_employee_name, cosine_similarity, image_base64 = query_result.iloc[0]
        if cosine_similarity > threshold:
            display_attendance_info(frame, result_employee_id, result_employee_name, cosine_similarity, facial_area)
            status_kehadiran = cek_kehadiran(result_employee_id, result_employee_name)
            if len(status_kehadiran) == 0:
                record_attendance(result_employee_id, result_employee_name)
                update_or_insert_employee(result_employee_id, result_employee_name, embedding, image_base64)
        else:
            info_text = 'Wajah tidak dikenali'
            x, y, w, h = facial_area[0], facial_area[1], facial_area[2], facial_area[3]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, info_text, (x - 30, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #cv2.imshow('Capture Face', frame)
            return frame
    else:
        print("Wajah tidak ditemukan. Coba lagi!")

# def main():
#     cap = cv2.VideoCapture(0)

#     while True:
#         ret, frame = cap.read()

#         cv2.imshow('Capture Face', frame)

#         try:
#             face_representation, base64_representation, facial_area = get_face_representation(frame)
#             compare_faces(frame, face_representation, facial_area)

#         except Exception as e:
#             #print(f'Error: {e}')
#             pass
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

if __name__ == "__main__":
    client = Client(host='localhost', port=9000, user='default', database='default')
    threshold = 0.825

    # main()