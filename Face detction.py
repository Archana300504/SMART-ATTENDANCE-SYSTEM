import cv2
import time
import threading
import numpy as np
from pymongo import MongoClient
from insightface.app import FaceAnalysis
from datetime import datetime
import uuid
import urllib.parse

import urllib.parse

CAMERAS = {
    "cam1": "rtsp://admin:{}@192.168.1.66:554/stream".format(urllib.parse.quote("sridevi@2004")),
    "cam2": "rtsp://admin:{}@192.168.1.65:554/stream".format(urllib.parse.quote("sridevi@2004"))
}

ENROLL_SNAPSHOTS = 5
CAPTURE_INTERVAL = 1.0       
RECENT_SEEN_TTL = 5.0         
EMBEDDING_REFRESH_INTERVAL = 30.0
FRAME_UPSCALE = 1.5          



client = MongoClient(
    "mongodb+srv://sih_ps_16:sih_ps_16@attendance-details.bpfxaxq.mongodb.net/"
    "?retryWrites=true&w=majority&appName=Attendance-details"
)
db = client["attendance_db"]
students_col = db["students"]
attendance_col = db["attendance"]
session_log_col = db["session_log"]
unknown_log_col = db["unknown_log"]


face_app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)


students_cache = []
students_lock = threading.Lock()
last_embedding_refresh = 0.0


def refresh_student_embeddings(force=False):
    global students_cache, last_embedding_refresh
    now = time.time()
    if not force and (now - last_embedding_refresh) < EMBEDDING_REFRESH_INTERVAL:
        return
    with students_lock:
        docs = list(students_col.find({}, {"reg_no":1, "name":1, "embedding":1}))
        cache = []
        for d in docs:
            if "embedding" in d and d["embedding"] is not None:
                cache.append({
                    "reg_no": d["reg_no"],
                    "name": d.get("name", ""),
                    "embedding": np.array(d["embedding"], dtype=float)
                })
        students_cache = cache
        last_embedding_refresh = now
        print(f"[cache] Loaded {len(students_cache)} student embeddings")

def cosine_similarity(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return -1.0
    return float(np.dot(a, b)/(na*nb))


def enroll_student():
    reg_no = input("Enter Register Number: ").strip()
    if not reg_no: return
    name = input("Enter Student Name: ").strip()
    department = input("Enter Department: ").strip()
    course = input("Enter Course: ").strip()
    password = input("Password: ").strip()

    print(f"\nStarting enrollment for {name} ({reg_no})")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Cannot open internal webcam")
        return

    embeddings = []
    captured = 0
    try:
        while captured < ENROLL_SNAPSHOTS:
            ret, frame = cap.read()
            if not ret: 
                time.sleep(0)
                continue

            
            h, w = frame.shape[:2]
            frame_up = cv2.resize(frame, (int(w*FRAME_UPSCALE), int(h*FRAME_UPSCALE)))

            faces = face_app.get(frame_up)
            cv2.putText(frame, f"Capturing {captured+1}/{ENROLL_SNAPSHOTS}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            cv2.imshow("Enrollment - Press q to cancel", frame)

            if len(faces) > 0:
                emb = faces[0].embedding
                embeddings.append(emb)
                captured += 1
                time.sleep(1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("Enrollment cancelled.")
        cap.release()
        cv2.destroyAllWindows()
        return

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings) == 0:
        print(" No faces captured. Enrollment failed.")
        return

    final_embedding = np.mean(embeddings, axis=0).tolist()
    students_col.insert_one({
        "reg_no": reg_no,
        "name": name,
        "department": department,
        "course": course,
        "password": password,
        "embedding": final_embedding,
        "enrolled_at": datetime.utcnow()
    })

    refresh_student_embeddings(force=True)
    print(f" Enrollment completed for {name} ({reg_no})\n")


def attendance_camera_worker(cam_url, cam_name):
    cap = cv2.VideoCapture(cam_url)


    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f" {cam_name} cannot open stream")
        return

    session_id = str(uuid.uuid4())
    window_name = f"Attendance - {cam_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    recently_seen = {}
    last_capture_time = 0.0
    last_unknown_notify = 0.0        
    UNKNOWN_NOTIFY_INTERVAL = 10       

    refresh_student_embeddings(force=True)
    print(f"[{cam_name}] started. Processing every {CAPTURE_INTERVAL}s")

    try:
        while True:

           
            for _ in range(5):
                cap.grab()

            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            now_time = time.time()
            if now_time - last_capture_time < CAPTURE_INTERVAL:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            last_capture_time = now_time

          
            h, w = frame.shape[:2]
            frame_up = cv2.resize(frame, (int(w * FRAME_UPSCALE), int(h * FRAME_UPSCALE)))

            faces = face_app.get(frame_up)
            total_faces = len(faces)
            recognized_ids = []
            recognized_count = 0
            unknown_count = 0
            now_dt = datetime.utcnow()
            date_str = now_dt.strftime("%Y-%m-%d")
            time_str = now_dt.strftime("%H:%M:%S")
            now_ts = now_time

            
            recently_seen = {k: v for k, v in recently_seen.items() if now_ts - v <= RECENT_SEEN_TTL}

            for face in faces:
                try:
                    emb = face.embedding
                except:
                    continue

                best_score = -1.0
                best_match = None

                with students_lock:
                    for s in students_cache:
                        score = cosine_similarity(emb, s["embedding"])
                        if score > best_score:
                            best_score = score
                            best_match = s

                
                try:
                    x1, y1, x2, y2 = (face.bbox / FRAME_UPSCALE).astype(int)
                    x1, y1 = max(x1, 0), max(y1, 0)
                    x2, y2 = min(x2, frame.shape[1]-1), min(y2, frame.shape[0]-1)
                except:
                    x1 = y1 = x2 = y2 = 0

                if best_score > 0.45 and best_match is not None:
                    reg_no = best_match["reg_no"]
                    name = best_match["name"]

                    if reg_no not in recently_seen:
                        attendance_col.update_one(
                            {"reg_no": reg_no, "date": date_str},
                            {"$set": {"name": name, "time": time_str, "attendance": 1, "camera": cam_name}},
                            upsert=True
                        )

                        print(f"[{cam_name}] ✔ Recognized: {name} ({reg_no}) score={best_score:.2f}")

                    recently_seen[reg_no] = now_ts
                    recognized_ids.append(reg_no)
                    recognized_count += 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{name} [{reg_no}] {best_score:.2f}", 
                                (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                else:
                    unknown_count += 1

                    
                    if now_ts - last_unknown_notify >= UNKNOWN_NOTIFY_INTERVAL:
                        unknown_log_col.insert_one({
                            "camera": cam_name,
                            "date": date_str,
                            "time": time_str,
                            "status": "Unknown Person",
                            "session_id": session_id,
                            "detected_at": datetime.utcnow()
                        })

                        print(f"[{cam_name}]Unknown detected score={best_score:.2f}")

                        last_unknown_notify = now_ts

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown",
                                (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

           
            session_log_col.insert_one({
                "session_id": session_id,
                "camera": cam_name,
                "date": date_str,
                "time": time_str,
                "total_faces": total_faces,
                "recognized": recognized_count,
                "unknown": unknown_count,
                "recognized_ids": recognized_ids,
                "logged_at": datetime.utcnow()
            })

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print(f"[{cam_name}] Interrupted.")

    finally:
        cap.release()
        cv2.destroyWindow(window_name)
        print(f"[{cam_name}] stopped session {session_id}.")


def main_menu():
    threads = {}
    try:
        while True:
            print("\n1. Enroll Student\n2. Start Attendance Camera 1\n3. Start Attendance Camera 2")
            print("4. Stop Camera 1\n5. Stop Camera 2\n6. Exit")
            choice = input("Enter choice: ").strip()
            if choice=="1": enroll_student()
            elif choice=="2":
                if "cam1" in threads and threads["cam1"].is_alive(): print("Cam1 running")
                else: 
                    t=threading.Thread(target=attendance_camera_worker,args=(CAMERAS["cam1"],"Camera 1"),daemon=True)
                    threads["cam1"]=t; t.start()
            elif choice=="3":
                if "cam2" in threads and threads["cam2"].is_alive(): print("Cam2 running")
                else:
                    t=threading.Thread(target=attendance_camera_worker,args=(CAMERAS["cam2"],"Camera 2"),daemon=True)
                    threads["cam2"]=t; t.start()
            elif choice=="4": print("Press 'q' in Cam1 window to stop")
            elif choice=="5": print("Press 'q' in Cam2 window to stop")
            elif choice=="6": break
            else: print("Invalid choice")
    except KeyboardInterrupt: pass
    print("Exiting...")
    time.sleep(0.1)

if __name__=="__main__":
    refresh_student_embeddings(force=True)
    main_menu()
