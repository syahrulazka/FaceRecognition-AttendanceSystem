import streamlit as st
import cv2
import av
import base64
import numpy as np
import pandas as pd
import threading
from PIL import Image
import time
from datetime import datetime, timedelta
from deepface import DeepFace
# from Recognition import recognition as rg
from clickhouse_driver import Client
from streamlit_webrtc import webrtc_streamer


def face_detection_transformer(frame):
    try:
        # Convert the BGR image to RGB
        rgb_frame = frame.to_ndarray(format="bgr24")

        rgb_frame = cv2.flip(rgb_frame, 1)

        # Detect faces using deepface
        result = DeepFace.represent(rgb_frame, model_name='Facenet512', enforce_detection=False, detector_backend='fastmtcnn')
        vector_embedding = result[0]['embedding']
        facial_area = list(result[0]['facial_area'].values())
        query_result = client.query_dataframe(
            f'SELECT employee_id, employee_name, 1 - cosineDistance({vector_embedding}, embedding) AS cosineSimilarity, image_base64 FROM employee ORDER BY cosineSimilarity DESC LIMIT 1'
        )


        # Draw bounding box around the detected face
        if not query_result.empty:
            # global result_employee_id, result_employee_name, cosine_similarity
            employee_id, employee_name, cosine_similarity, image_base64 = query_result.iloc[0]
            x, y, w, h = facial_area
            if cosine_similarity > threshold:
                #Draw bounding box around the detected face
                #x, y, w, h = facial_area
                rgb_frame = cv2.rectangle(rgb_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                rgb_frame = cv2.putText(rgb_frame, employee_name, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                rgb_frame = cv2.putText(rgb_frame, 'Absensi Berhasil', (50, 50), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 255, 0) , 2) 
                cv2.putText(rgb_frame, f'Similarity: {format(cosine_similarity, ".2%")}', (x, y + h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                try:
                    catat_yang_mencoba_masuk(employee_id, employee_name,cosine_similarity)
                except Exception as e:
                    print(f'error: {e}')    

                try:
                    if len(cek_kehadiran(employee_id, employee_name)) == 0:
                        # Jika belum hadir, rekam kehadiran
                        record_attendance(employee_id, employee_name)
        
                        # Update atau insert data employee
                        update_or_insert_employee(employee_id, employee_name, vector_embedding, image_base64)

                except Exception as e:
                    print(f"Error during attendance recording or employee data update/insert: {e}")
            else:
                cv2.rectangle(rgb_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(rgb_frame, format="bgr24")

    except Exception as e:
        #st.error(f'Error: {e}')
        pass
        return av.VideoFrame.from_ndarray(cv2.flip(frame.to_ndarray(format="bgr24"), 1), format="bgr24")
    
def cek_kehadiran(employee_id, employee_name):
    today_date = (datetime.now()).strftime('%Y-%m-%d')
    query = f"SELECT * FROM kehadiran WHERE employee_id = '{employee_id}' AND employee_name = '{employee_name}' AND waktu_kehadiran >= '{today_date}'"
    hadir = client.execute(query)
    return hadir # jika len(hadir) = 0 maka orang tsb belum hadir

def record_attendance(employee_id, employee_name):
    current_time = (datetime.now() + timedelta(hours=7)).strftime('%Y-%m-%d %H:%M:%S')
    client.execute(
        f'INSERT INTO kehadiran (employee_id, employee_name, waktu_kehadiran) VALUES ({employee_id}, \'{employee_name}\', \'{current_time}\')'
    )

def update_or_insert_employee(employee_id, employee_name, embedding, base64_representation):
    client.execute(
        f'INSERT INTO employee (employee_id, employee_name, embedding, image_base64) '
        f'VALUES ({employee_id}, \'{employee_name}\', {embedding}, \'{base64_representation}\') '
    )

def catat_yang_mencoba_masuk(employee_id, employee_name, cosine_similarity):
    current_time = (datetime.now() + timedelta(hours=7)).strftime('%Y-%m-%d %H:%M:%S')
    try:
        client.execute(
            f'INSERT INTO mencobamasuk (employee_id, employee_name, waktu, cosine_similarity) VALUES ({employee_id}, \'{employee_name}\', \'{current_time}\', {cosine_similarity})'
        )
    except Exception as e:
        print(e)

def get_last_entry_from_clickhouse():
    try:
        # Mendapatkan data terakhir
        query = f"""SELECT
    m.employee_id,
    m.employee_name,
    m.cosine_similarity,
    k.waktu_kehadiran
FROM
    (SELECT employee_id, employee_name, cosine_similarity
     FROM mencobamasuk
     ORDER BY waktu DESC
     LIMIT 1) AS m
JOIN
    kehadiran k ON m.employee_id = k.employee_id
WHERE toDate(k.waktu_kehadiran) = today() LIMIT 1
    """
        ### WHERE toDate(k.waktu_kehadiran) = today() LIMIT 1
        result = client.execute(query)
        return result[0] #if result else None

    except Exception as e:
        #st.error(f'Error during ClickHouse query: {e}')
        pass
        #return None

def display_info_streamlit():
    result_container = st.empty()

    while True:
        # Mendapatkan data terakhir
        last_entry = get_last_entry_from_clickhouse()

        if last_entry:
            employee_id, employee_name, cosine_similarity, waktu_kehadiran = last_entry
            waktu_kehadiran  = waktu_kehadiran - timedelta(hours=7)
            data = {'Employee ID': [employee_id], 'Employee Name': [employee_name], 'Waktu Absensi Hari Ini': [waktu_kehadiran], 'Status': 'Hadir'}
            df = pd.DataFrame(data)
            result_container.table(df)
        else:
            #result_container.write("Wajah tidak dikenali")
            pass
        # Tunggu beberapa detik sebelum memperbarui kembali
        time.sleep(0.5)

def absensi():
    st.title("Absensi - Face Recognition")
    st.subheader("Silakan lakukan absensi di sini!")
    
    webrtc_streamer(
        key="example",
        video_frame_callback=face_detection_transformer
    )
    st.subheader('Keterangan:')
    display_info_streamlit()    


if __name__ == "__main__":
    client = Client(host='localhost', port=9000, user='default', database='default')
    threshold = 0.825
    absensi()



