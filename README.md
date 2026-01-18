# SMART-ATTENDANCE-SYSTEM
This is a Face detection based attendance monitoring system with the integration of the automated predictive analysis system in a web page

#  Real-Time Face Recognition Attendance System

A robust **AI-powered attendance management system** that uses **real-time face recognition** to automatically mark student attendance using **IP cameras**, **InsightFace**, **OpenCV**, and **MongoDB**.  

# Problem Statement
Traditional attendance systems are time-consuming, error-prone, and vulnerable to proxy attendance.  
This project addresses these challenges by providing a **contactless, automated, and scalable face recognition–based attendance solution**.

# Key Features
- 🔍 Real-time face detection and recognition
- 🧠 High-accuracy face embeddings using **InsightFace (buffalo_l)**
- 🎥 Multi-camera support via RTSP streams
- 🧑‍🎓 Student enrollment with facial embeddings
- ⏱️ Duplicate attendance prevention using time-based logic
- 🚨 Unknown person detection and logging
- 🗂️ Centralized MongoDB database storage
- 🧵 Multi-threaded camera processing


# Technology Stack
- **Programming Language:** Python 3
- **Computer Vision:** OpenCV
- **Face Recognition:** InsightFace
- **Database:** MongoDB Atlas
- **Numerical Computing:** NumPy
- **Concurrency:** Python Threading
- **Cameras:** RTSP IP Cameras / CCTV

# Working Methodology
#  Student Enrollment

-Captures multiple face snapshots
-Generates facial embeddings
-Stores averaged embedding in MongoDB

# Attendance Processing

-Reads frames from IP cameras
-Detects and recognizes faces
-Matches embeddings using cosine similarity
-Marks attendance when similarity exceeds threshold

# Duplicate Prevention

-Recently recognized students are ignored for a defined time window
-Prevents multiple attendance entries for the same person

# Unknown Person Detection

-Faces not matching the database are labeled Unknown
-Unknown entries are logged periodically for security review


# Database Design
# students

-reg_no
-name
-department
-course
-embedding
-enrolled_at

# attendance

-reg_no
-name
-date
-time
-camera
-attendance

# session_log

-session_id
-camera
-total_faces
-recognized
-unknown
-recognized_ids

# unknown_log

-camera
-date
-time
-status
-session_id


# How to Run the Project
1. Install Dependencies
pip install opencv-python numpy pymongo insightface onnxruntime

2. Run the Application
python main.py

3. User Menu
1. Enroll Student
2. Start Attendance Camera 1
3. Start Attendance Camera 2
4. Stop Camera 1
5. Stop Camera 2
6. Exit

# Output

Live camera feed with bounding boxes
Recognized faces labeled with name and register number
Attendance records stored automatically
Session-wise analytics
Unknown face logs for security auditing



# Future Enhancements

-Web-based dashboard
-Role-based authentication
-Mask detection
-Attendance analytics and reports
-Cloud deployment
-LMS / ERP integration

# Author
Archana Devi P M

