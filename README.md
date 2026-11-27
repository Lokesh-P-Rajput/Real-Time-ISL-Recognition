# 🤟 Real-Time Indian Sign Language Recognition and Translation System

## 1. Introduction

The Real-Time Indian Sign Language (ISL) Recognition and Translation System is a Python-based web application designed to reduce communication barriers between people who use Indian Sign Language and people who do not understand sign language.

The application uses Computer Vision and Machine Learning to recognize hand gestures through a standard webcam and convert them into readable text and speech.

## 2. Project Objectives

- Recognize basic Indian Sign Language gestures in real time.
- Detect hand landmarks using MediaPipe.
- Classify hand gestures using Machine Learning.
- Convert recognized gestures into text.
- Generate voice output using Text-to-Speech.
- Maintain translation history using SQLite.

## 3. Features

- Real-time webcam input.
- Detection of 21 hand landmarks.
- Machine Learning-based gesture recognition.
- Text translation.
- Offline voice output.
- Translation history.
- Streamlit-based web interface.
- Start and Stop controls.

## 4. Technologies Used

- Python
- OpenCV
- MediaPipe
- Scikit-learn
- XGBoost
- NumPy
- Pandas
- Streamlit
- pyttsx3
- SQLite

## 5. System Workflow

1. Capture live video using a webcam.
2. Detect hands using MediaPipe.
3. Extract 21 hand landmarks.
4. Normalize landmark coordinates.
5. Extract numerical features.
6. Predict the gesture using the trained ML model.
7. Display the recognized gesture as text.
8. Convert the text into speech.
9. Save the translation history in SQLite.

## 6. Supported Gestures

The initial prototype targets approximately 15–20 isolated daily-use gestures, including:

- Hello
- Thank You
- Yes
- No
- Help
- Sorry
- Please
- Stop
- Water
- Food
- Welcome
- Good
- Bad
- Come
- Go

## 7. Installation and Setup

### Prerequisites

- Python installed on Windows 10/11.
- Working webcam.
- VS Code or another Python IDE.
- Required project files and trained model.

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Real-Time-ISL-Recognition.git
```

### Step 2: Open the project folder

```bash
cd Real-Time-ISL-Recognition
```

### Step 3: Create a virtual environment

```bash
python -m venv venv
```

### Step 4: Activate the environment

Windows:

```bash
venv\Scripts\activate
```

### Step 5: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the application

```bash
streamlit run app.py
```

Replace `app.py` with the actual name of your main application file if different.

### Step 7: Open the application

Open the local URL displayed in your terminal by Streamlit and allow webcam access if prompted.

## 8. How to Use

1. Open the application in your browser.
2. Click the Start button.
3. Sit in a well-lit place.
4. Show a supported hand gesture clearly in front of the webcam.
5. View the detected gesture and translated text.
6. Listen to the generated voice output.
7. Check the translation history.
8. Click Stop to end the session.

## 9. Project Limitations

- Currently focuses on approximately 15–20 isolated gestures.
- Does not provide complete ISL vocabulary translation.
- Continuous sentence recognition is outside the current scope.
- Two-handed and dynamic gestures are outside the current scope.
- Recognition may be affected by lighting, background and signing style.

## 10. Future Scope

- Expand the gesture vocabulary.
- Add dynamic gesture recognition.
- Implement continuous sentence translation.
- Improve recognition accuracy.
- Explore advanced deep-learning models.

## 11. Contributors

**Lokesh Rajput**  
**Ayaj Kha Shaikh**

Master of Computer Applications  
Medicaps University, Indore

## 12. Project Category

Academic Project | Artificial Intelligence | Machine Learning | Computer Vision | Python