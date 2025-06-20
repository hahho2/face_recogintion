import streamlit as st
import cv2
import pandas as pd
import numpy as np
import face_recognition
import os
import pickle
from datetime import datetime, time
from PIL import Image

# --- Configuration & Setup ---
DATA_DIR = "data"
LOG_DIR = "logs"
ENCODINGS_PATH = os.path.join(DATA_DIR, "faces_encodings.pkl")
NAMES_PATH = os.path.join(DATA_DIR, "known_names.pkl")
CASCADE_PATH = os.path.join(DATA_DIR, 'haarcascade_frontalface_default.xml') # Keep for initial detection if preferred
FACES_TO_CAPTURE = 80 # Capture fewer faces for quicker registration in web app

# --- Ensure Directories Exist ---
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- Load Haar Cascade (Optional but can speed up initial detection) ---
try:
    facedetect = cv2.CascadeClassifier(CASCADE_PATH)
    if facedetect.empty():
        st.error(f"Error loading Haar Cascade from {CASCADE_PATH}. Face detection might be slower or fail.")
        # Fallback or proceed using only face_recognition's HOG detector
except Exception as e:
    st.error(f"Exception loading Haar Cascade: {e}")
    # Fallback or proceed

# --- Helper Functions ---

def load_known_faces():
    """Loads known face encodings and names."""
    known_face_encodings = []
    known_names = []
    if os.path.exists(ENCODINGS_PATH) and os.path.exists(NAMES_PATH):
        try:
            with open(ENCODINGS_PATH, 'rb') as f:
                known_face_encodings = pickle.load(f)
            with open(NAMES_PATH, 'rb') as f:
                known_names = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, FileNotFoundError) as e:
            st.error(f"Error loading existing face data: {e}. Starting fresh.")
            # Optionally delete corrupted files here
            if os.path.exists(ENCODINGS_PATH): os.remove(ENCODINGS_PATH)
            if os.path.exists(NAMES_PATH): os.remove(NAMES_PATH)
            return [], [] # Return empty lists
        except Exception as e:
            st.error(f"An unexpected error occurred loading face data: {e}")
            return [], []
    return known_face_encodings, known_names

def save_known_faces(new_encoding, new_name_roll):
    """Appends a new face encoding and name to the stored data."""
    known_face_encodings, known_names = load_known_faces()

    # Check if name_roll already exists
    if new_name_roll in known_names:
        st.warning(f"'{new_name_roll}' already exists. Not adding duplicate.")
        return False # Indicate not added

    known_face_encodings.append(new_encoding)
    known_names.append(new_name_roll)

    try:
        with open(ENCODINGS_PATH, 'wb') as f:
            pickle.dump(known_face_encodings, f)
        with open(NAMES_PATH, 'wb') as f:
            pickle.dump(known_names, f)
        return True # Indicate success
    except Exception as e:
        st.error(f"Error saving face data: {e}")
        return False # Indicate failure

def mark_attendance(name, roll):
    """Marks attendance in a CSV file for the current date."""
    ts = datetime.now()
    date = ts.strftime("%d-%m-%Y")
    timestamp = ts.strftime("%H:%M:%S")
    file_path = os.path.join(LOG_DIR, f"Attendance_{date}.csv")

    try:
        if not os.path.isfile(file_path):
            df = pd.DataFrame(columns=["Name", "Roll", "Time"])
        else:
            # Read existing, handling potential empty file or parsing errors
            try:
                df = pd.read_csv(file_path)
                if df.empty and os.path.getsize(file_path) > 0: # Header only?
                     df = pd.DataFrame(columns=["Name", "Roll", "Time"])
            except pd.errors.EmptyDataError:
                 df = pd.DataFrame(columns=["Name", "Roll", "Time"])
            except Exception as read_e:
                 st.error(f"Error reading attendance file {file_path}: {read_e}")
                 df = pd.DataFrame(columns=["Name", "Roll", "Time"]) # Fallback


        # Ensure correct columns exist even if file was malformed/empty
        if not all(col in df.columns for col in ["Name", "Roll", "Time"]):
             df = pd.DataFrame(columns=["Name", "Roll", "Time"])

        # Convert Roll column to string for reliable comparison if it exists and has data
        if 'Roll' in df.columns and not df['Roll'].empty:
            df['Roll'] = df['Roll'].astype(str)


        # Check if already marked today
        # Compare roll as string to avoid int/str issues
        if not ((df['Name'] == name) & (df['Roll'] == str(roll))).any():
            new_entry = pd.DataFrame([[name, str(roll), timestamp]], columns=["Name", "Roll", "Time"])
            df = pd.concat([df, new_entry], ignore_index=True)
            df.to_csv(file_path, index=False)
            st.success(f"Attendance marked for {name} ({roll}) at {timestamp}")
            return True # Marked
        else:
            st.info(f"{name} ({roll}) already marked today.")
            return False # Already marked
    except Exception as e:
        st.error(f"Error updating attendance file {file_path}: {e}")
        return False # Error occurred


# --- Streamlit App Interface ---
st.set_page_config(page_title="Face Recognition Attendance", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .title-text {
        font-size: 36px;
        font-weight: 600;
        color: #ffffff;
        text-align: center;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        background: linear-gradient(145deg, #2c2c2c, #1a1a1a);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background: linear-gradient(145deg, #3a3a3a, #2c2c2c);
        color: #ffffff;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        background: linear-gradient(145deg, #444444, #363636);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    div[data-testid="stSidebarNav"] {
        background: linear-gradient(145deg, #2c2c2c, #1a1a1a);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .sidebar-text {
        color: #ffffff;
        font-size: 22px;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 600;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .stSelectbox > div > div {
        background-color: #2c2c2c;
        border-radius: 8px;
        color: #ffffff;
        border: 1px solid #444444;
    }
    .stTextInput > div > div > input {
        background-color: #2c2c2c;
        border-radius: 8px;
        color: #ffffff;
        border: 1px solid #444444;
    }
    div.stDataFrame {
        background: linear-gradient(145deg, #2c2c2c, #1a1a1a);
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# Title with custom styling
st.markdown('<p class="title-text">🎓 Smart Attendance System</p>', unsafe_allow_html=True)

# --- Sidebar Navigation with better styling ---
st.sidebar.markdown('<p class="sidebar-text">Navigation</p>', unsafe_allow_html=True)
menu = ["📸 Take Attendance", "➕ Add New Face", "📊 View Attendance Log"]
choice = st.sidebar.selectbox("", menu)

# --- Load Known Faces Globally ---
if 'known_face_encodings' not in st.session_state or 'known_names' not in st.session_state:
    st.session_state.known_face_encodings, st.session_state.known_names = load_known_faces()
    st.sidebar.info(f"✅ System loaded with {len(st.session_state.known_names)} known faces")

# --- Mode: Take Attendance ---
if "📸 Take Attendance" in choice:
    st.header("📸 Take Attendance")
    col1, col2 = st.columns([2,1])
    with col1:
        st.write("Position your face in front of the camera")
        img_placeholder = st.empty()
    with col2:
        start_button = st.button("▶️ Start Recognition")
        stop_button = st.button("⏹️ Stop Recognition")

    if 'attendance_running' not in st.session_state:
        st.session_state.attendance_running = False

    if start_button:
        st.session_state.attendance_running = True
        st.session_state.marked_today = set() # Keep track of who was marked in this session

    if stop_button:
        st.session_state.attendance_running = False

    if st.session_state.attendance_running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam.")
            st.session_state.attendance_running = False
        else:
            st.info("Webcam started. Looking for faces...")
            while st.session_state.attendance_running:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to grab frame from webcam.")
                    break

                # --- Face Recognition Logic ---
                # Resize frame for faster processing (optional)
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5) # Half size
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Find all face locations and encodings in the current frame
                face_locations = face_recognition.face_locations(rgb_small_frame) # model='cnn' is more accurate but slower
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                recognized_this_frame = False
                for face_encoding, face_location in zip(face_encodings, face_locations):
                    # Compare with known faces
                    matches = face_recognition.compare_faces(st.session_state.known_face_encodings, face_encoding, tolerance=0.5) # Adjust tolerance as needed
                    name_roll = "Unknown"
                    roll = "N/A"

                    face_distances = face_recognition.face_distance(st.session_state.known_face_encodings, face_encoding)
                    if len(face_distances) > 0: # Check if there are known faces to compare against
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name_roll = st.session_state.known_names[best_match_index]
                            try:
                                # Robust splitting assuming "Name_Roll" format
                                parts = name_roll.split('_')
                                name = "_".join(parts[:-1]) if len(parts) > 1 else name_roll # Handle names with underscores
                                roll = parts[-1] if len(parts) > 1 else "N/A"
                            except:
                                name = name_roll # Fallback if split fails
                                roll = "N/A"

                            # Mark attendance only once per session for a person
                            if name_roll not in st.session_state.get('marked_today', set()):
                                if mark_attendance(name, roll):
                                    st.session_state.marked_today.add(name_roll)
                                    recognized_this_frame = True # Stop after one successful mark per frame loop? Or allow multiple?

                    # --- Draw Bounding Box on Original Frame ---
                    top, right, bottom, left = face_location
                    # Scale back up face locations since they were detected on the small frame
                    top *= 2
                    right *= 2
                    bottom *= 2
                    left *= 2
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0) if name_roll != "Unknown" else (0, 0, 255), 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0) if name_roll != "Unknown" else (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name_roll, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

                # Display the resulting frame
                img_placeholder.image(frame, channels="BGR")

                # Option: Stop automatically after successful recognition
                # if recognized_this_frame:
                #     st.session_state.attendance_running = False
                #     st.success("Recognition complete for this run.")

                # Check if the stop button was pressed during the loop (Streamlit reruns)
                # This check might not be strictly necessary if stop button logic works correctly
                if not st.session_state.get('attendance_running', False):
                     break # Exit loop if state changed

            cap.release()
            if st.session_state.get('attendance_running', False) == False : # Clear image if stopped
                 img_placeholder.empty()
                 st.info("Webcam stopped.")


# --- Mode: Add New Face ---
elif "➕ Add New Face" in choice:
    st.header("➕ Register New Face")
    
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("👤 Enter Name")
    with col2:
        roll_str = st.text_input("🔢 Enter Roll Number")
    
    st.info(f"ℹ️ System will capture {FACES_TO_CAPTURE} face samples")
    
    if st.button("📸 Start Capturing"):
        if not name or not roll_str:
            st.warning("Please enter both Name and Roll Number.")
        else:
            name_roll = f"{name}_{roll_str}"
            if name_roll in st.session_state.known_names:
                 st.warning(f"'{name_roll}' already exists. Choose a different name or roll number.")
            else:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Could not open webcam.")
                else:
                    st.info("Webcam started. Look at the camera.")
                    img_placeholder_add = st.empty()
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    face_encodings_list = []
                    faces_captured = 0
                    max_attempts = FACES_TO_CAPTURE * 5 # Try a bit harder to get samples

                    for i in range(max_attempts):
                        if faces_captured >= FACES_TO_CAPTURE:
                            break # Got enough samples

                        ret, frame = cap.read()
                        if not ret:
                            st.warning("Failed to grab frame.")
                            time.sleep(0.1) # Short pause
                            continue

                        # Convert to RGB for face_recognition
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Detect faces (using face_recognition detector is better here)
                        face_locations = face_recognition.face_locations(rgb_frame) # Use default HOG model

                        if len(face_locations) == 1: # Ensure only one face is clearly visible
                            # Get encoding for the single detected face
                            current_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                            if current_encodings: # Check if encoding was successful
                                face_encodings_list.append(current_encodings[0])
                                faces_captured += 1
                                progress = int((faces_captured / FACES_TO_CAPTURE) * 100)
                                progress_bar.progress(progress)
                                status_text.text(f"Captured: {faces_captured}/{FACES_TO_CAPTURE}")

                                # Draw rectangle on the display frame (BGR)
                                top, right, bottom, left = face_locations[0]
                                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                            else:
                                status_text.text(f"Could not encode face {faces_captured+1}/{FACES_TO_CAPTURE}. Try different angle/lighting.")
                                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2) # Red box if encoding failed

                        elif len(face_locations) > 1:
                             status_text.text("Multiple faces detected. Please ensure only one person is visible.")
                        else:
                             status_text.text("No face detected. Please position yourself clearly.")


                        # Display frame
                        img_placeholder_add.image(frame, channels="BGR")

                        # Allow Streamlit to process events slightly, prevent freezing
                        # cv2.waitKey(1) # Not effective in Streamlit
                        # time.sleep(0.05) # Small delay

                    cap.release()
                    img_placeholder_add.empty() # Clear image after capture
                    progress_bar.empty()
                    status_text.empty()

                    if faces_captured >= FACES_TO_CAPTURE:
                        # Average the encodings (or just use the first one if averaging is complex/not needed)
                        # Averaging can sometimes produce a more robust representation
                        # mean_encoding = np.mean(np.array(face_encodings_list), axis=0)

                        # Simpler: Just use the first captured encoding or a selection
                        # For simplicity, let's just use the first good one.
                        # In a real app, you might want quality checks or averaging.
                        if face_encodings_list:
                            representative_encoding = face_encodings_list[0] # Or np.mean(face_encodings_list, axis=0)
                            if save_known_faces(representative_encoding, name_roll):
                                st.success(f"Successfully added '{name_roll}'.")
                                # Reload known faces in session state
                                st.session_state.known_face_encodings, st.session_state.known_names = load_known_faces()
                                st.sidebar.info(f"Loaded {len(st.session_state.known_names)} known faces.") # Update sidebar count
                            else:
                                st.error(f"Failed to save '{name_roll}'. Check logs or permissions.")
                        else:
                             st.error("Could not capture any valid face encodings.")

                    else:
                        st.error(f"Failed to capture enough face samples ({faces_captured}/{FACES_TO_CAPTURE}). Please try again.")

# --- Mode: View Attendance Log ---
elif "📊 View Attendance Log" in choice:
    st.header("📊 Attendance Records")
    
    log_files = sorted([f for f in os.listdir(LOG_DIR) if f.startswith("Attendance_") and f.endswith(".csv")], reverse=True)

    if not log_files:
        st.info("No attendance logs found.")
    else:
        selected_log = st.selectbox("Select Log Date", log_files)
        if selected_log:
            file_path = os.path.join(LOG_DIR, selected_log)
            try:
                df_log = pd.read_csv(file_path)
                st.dataframe(df_log)

                # Option to download
                st.download_button(
                    label="Download CSV",
                    data=df_log.to_csv(index=False).encode('utf-8'),
                    file_name=selected_log,
                    mime='text/csv',
                )
            except pd.errors.EmptyDataError:
                 st.warning(f"Log file '{selected_log}' is empty.")
            except Exception as e:
                st.error(f"Error reading log file {selected_log}: {e}")

# --- Footer with better styling ---
st.sidebar.markdown("---")
st.sidebar.markdown("""
    <div style='text-align: center; color: #666;'>
        <h4>About</h4>
        <p>Smart Attendance System v1.0</p>
        <p>Using AI-Powered Face Recognition</p>
    </div>
""", unsafe_allow_html=True)