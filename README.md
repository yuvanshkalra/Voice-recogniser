

# SSO Consultants Voice Recogniser 🎙️

### Machine Learning-Based Actor & Actress Voice Identification

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest-green)](https://scikit-learn.org/)
[![Audio Processing](https://img.shields.io/badge/Audio-Librosa-orange)](https://librosa.org/)
[![Firebase](https://img.shields.io/badge/Cloud-Firebase-FFCA28?logo=firebase)](https://firebase.google.com/)

**SSO Consultants Voice Recogniser** is a machine learning-based speaker identification application built with Python and Streamlit. It identifies known actors and actresses from their voice recordings using Mel-Frequency Cepstral Coefficients (MFCCs) and a Random Forest classifier.

The application supports live voice recording and WAV file uploads, stores voice samples in Firebase Cloud Storage, maintains actor metadata in Cloud Firestore, and provides an administrator interface for collecting training data, retraining the model, and managing actor information.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [How It Works](#-how-it-works)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Running the Application](#-running-the-application)
- [Using the Application](#-using-the-application)
- [Model Evaluation](#-model-evaluation)
- [Firebase Data Organization](#-firebase-data-organization)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🚀 Overview

Identifying a speaker from an audio recording is a speaker recognition problem in which a machine learning model learns patterns from known speakers and predicts which enrolled speaker most closely matches a new recording.

This project applies audio signal processing and supervised machine learning to recognize registered actors and actresses from their voice samples.

The application combines:

- Audio recording and playback
- MFCC-based feature extraction
- Random Forest classification
- Model training and evaluation
- Firebase-based audio storage
- Cloud Firestore metadata management
- A Streamlit interface with administrator and user roles

### 🎯 Project Objectives

- Build a voice-based speaker identification system.
- Extract useful acoustic features from recorded speech.
- Train a supervised machine learning model on labeled voice samples.
- Identify known speakers from new audio recordings.
- Store voice datasets and trained model artifacts in Firebase.
- Display relevant metadata for recognized actors and actresses.
- Evaluate classification performance using standard machine learning metrics.

---

## ✨ Key Features

### 1. 🎙️ Live Voice Recognition

Users can record audio directly through the application's microphone interface.

The recorded audio is processed using the trained machine learning model, and the predicted speaker's name is displayed along with available metadata.

### 2. 📂 WAV File Recognition

- Upload a WAV audio file.
- Preview the uploaded recording.
- Extract MFCC features from the audio.
- Predict the identity of the most likely enrolled speaker.
- Display additional information associated with the predicted identity.

The current file-upload interface accepts WAV files only.

### 3. 🧠 MFCC Feature Extraction

The application uses Librosa to extract 13 Mel-Frequency Cepstral Coefficients (MFCCs) from audio recordings.

MFCCs represent important characteristics of the short-term spectral envelope of audio signals and are commonly used in speech and speaker recognition systems.

The application averages the MFCC values across time to produce a fixed-length feature vector for each recording.

### 4. 🌲 Random Forest Classification

A Random Forest classifier learns patterns from the extracted audio features and their corresponding speaker labels.

The model is configured with:

- 150 decision trees (`n_estimators=150`)
- Maximum tree depth of 10
- Minimum of 5 samples per leaf
- Balanced class weights
- A fixed random state for reproducibility

The trained model predicts the speaker associated with an input recording.

### 5. ☁️ Firebase Cloud Integration

The application uses Firebase for persistent storage and metadata management.

**Firebase Cloud Storage**
- Stores WAV training recordings.
- Stores the trained Random Forest model.
- Stores the speaker ID-to-label mapping.

**Cloud Firestore**
- Stores actor and actress metadata.
- Retrieves information associated with predicted speakers.
- Supports editing and updating existing metadata.

### 6. 👨‍💼 Administrator Dashboard

The administrator interface provides three primary workflows.

**Add Speaker Data**
- Enter the actor's or actress's name.
- Record five voice samples.
- Enter age, height, total films, hit films, and industry.
- Upload the recordings to Firebase.
- Save metadata in Firestore.
- Retrain the recognition model.

**Retrain Model**
- Reload available voice samples from Firebase.
- Extract audio features.
- Train a new Random Forest model.
- Evaluate the model on a held-out test set.
- Save and upload the trained model and label mapping.

**View/Update Actor Details**
- View existing actor and actress records.
- Edit metadata in an interactive table.
- Save changes to Cloud Firestore.

### 7. 📊 Model Evaluation and Visualization

During model training, the application displays:

- Training and testing sample counts
- Test-set accuracy
- Classification report
- Confusion matrix

These outputs help assess how well the classifier distinguishes between the enrolled speakers.

### 8. 🔐 Role-Based Application Interface

The application separates access into two roles:

| Role | Capabilities |
|---|---|
| Administrator | Record training data, manage metadata, retrain the model |
| User | Identify speakers using uploaded WAV files or live recordings |

Login credentials are configured through Streamlit secrets.

---

## ⚙️ How It Works

```text
          ┌──────────────────────────┐
          │ Administrator Records    │
          │ Speaker Voice Samples    │
          └────────────┬─────────────┘
                       │
                       ▼
          ┌──────────────────────────┐
          │ Firebase Cloud Storage   │
          │ Labeled WAV Recordings   │
          └────────────┬─────────────┘
                       │
                       ▼
          ┌──────────────────────────┐
          │ Audio Preprocessing      │
          │ Librosa                  │
          └────────────┬─────────────┘
                       │
                       ▼
          ┌──────────────────────────┐
          │ MFCC Feature Extraction  │
          │ 13 Coefficients          │
          └────────────┬─────────────┘
                       │
                       ▼
          ┌──────────────────────────┐
          │ Random Forest Classifier │
          │ Train / Test Split       │
          └────────────┬─────────────┘
                       │
                       ▼
          ┌──────────────────────────┐
          │ Save Model and Labels    │
          │ Firebase Storage         │
          └──────────────────────────┘


          ───── Recognition Workflow ─────

          ┌──────────────────────────┐
          │ Live Recording or WAV    │
          │ File Upload              │
          └────────────┬─────────────┘
                       │
                       ▼
          ┌──────────────────────────┐
          │ Extract 13 MFCC Features │
          └────────────┬─────────────┘
                       │
                       ▼
          ┌──────────────────────────┐
          │ Trained Model Prediction │
          └────────────┬─────────────┘
                       │
                       ▼
          ┌──────────────────────────┐
          │ Predicted Speaker Name   │
          │ + Firestore Metadata     │
          └──────────────────────────┘
```

### Workflow Explanation

1. An administrator records five voice samples for a speaker.
2. The recordings are saved temporarily and uploaded to Firebase Cloud Storage.
3. Speaker metadata is saved in Cloud Firestore.
4. The application downloads available WAV recordings and extracts MFCC features.
5. The labeled feature vectors are divided into training and testing sets.
6. A Random Forest classifier is trained on the training set.
7. The model is evaluated using the test set.
8. The trained classifier and speaker-label mapping are serialized using Python's `pickle` module and uploaded to Firebase.
9. Users upload a WAV file or record their voice live.
10. The application extracts features and predicts the speaker's identity.
11. The speaker's available metadata is retrieved from Firestore and displayed.

---

## 🧪 Machine Learning Pipeline

### Audio Preprocessing

The application loads audio through Librosa at a standardized sample rate of 44,100 Hz.

It checks for recordings that are too short or nearly silent before feature extraction.

### Feature Extraction

The application computes 13 MFCCs:

\[
\mathbf{x} = [c_1,c_2,\ldots,c_{13}]
\]

The coefficients are averaged over the recording's time frames, producing a fixed-length representation of each voice sample.

### Dataset Construction

Voice recordings are organized by speaker name in Firebase Storage.

Each recording is assigned a numeric class label corresponding to its speaker.

### Training and Testing

The dataset is split into:

- 80% training data
- 20% testing data

The split uses `random_state=42` and stratification to preserve speaker proportions.

At least two distinct speakers are required for training. The application also checks whether each speaker has sufficient samples for the split and warns when the dataset is small.

### Classification

The trained Random Forest model predicts a numeric speaker label, which is mapped back to the corresponding speaker name.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Streamlit | Interactive web interface |
| NumPy | Numerical operations and feature arrays |
| SciPy | Audio and WAV-related utilities |
| Librosa | Audio loading and MFCC feature extraction |
| Scikit-learn | Random Forest, dataset splitting, and evaluation |
| Firebase Admin SDK | Cloud Storage and Firestore access |
| Firebase Cloud Storage | Audio samples and serialized model storage |
| Cloud Firestore | Speaker metadata management |
| pandas | Tabular metadata handling |
| Matplotlib | Model evaluation visualizations |
| Seaborn | Confusion matrix visualization |
| streamlit-audiorec | Browser-based audio recording |
| pickle | Model and label serialization |

---

## 📂 Project Structure

```text
Voice-recogniser/
│
├── app.py
│   ├── Firebase initialization
│   ├── Audio upload and recording
│   ├── MFCC feature extraction
│   ├── Dataset preparation
│   ├── Random Forest model training
│   ├── Model evaluation
│   ├── Speaker prediction
│   ├── Actor metadata management
│   └── Streamlit user interface
│
├── requirements.txt
│   └── Python dependencies
│
├── sso_logo.png
│   └── Application branding
│
├── README.md
│   └── Project documentation
│
├── .streamlit/
│   └── secrets.toml             # Create locally; do not commit
│
└── temp_recordings/             # Created automatically at runtime
    ├── Temporary WAV recordings
    ├── Downloaded model files
    └── Temporary label mapping
```

The `.streamlit/secrets.toml` and `temp_recordings/` entries describe local configuration and runtime files, not additional tracked files in the repository.

---

## 💻 Prerequisites

Before running the application, ensure that you have:

- Python installed
- pip package manager
- A microphone for live voice recording
- A Firebase project
- Firebase Admin SDK service-account credentials
- A configured Firebase Storage bucket
- Cloud Firestore enabled
- Appropriate access permissions for the Firebase resources

---

## 📥 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yuvanshkalra/Voice-recogniser.git
```

### 2. Navigate to the Project Directory

```bash
cd Voice-recogniser
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The project uses `streamlit-audiorec` for browser-based microphone recording. Ensure the component works with your installed Streamlit version and browser.

---

## 🔑 Configuration

The application requires Firebase credentials and login credentials through Streamlit secrets.

Create this file:

```text
.streamlit/secrets.toml
```

### Firebase Configuration

```toml
[firebase]
service_account_json = '''
{
  "type": "service_account",
  "project_id": "YOUR_PROJECT_ID",
  "private_key_id": "YOUR_PRIVATE_KEY_ID",
  "private_key": "YOUR_PRIVATE_KEY",
  "client_email": "YOUR_SERVICE_ACCOUNT_EMAIL",
  "client_id": "YOUR_CLIENT_ID",
  "token_uri": "https://oauth2.googleapis.com/token"
}
'''

storage_bucket = "YOUR_FIREBASE_STORAGE_BUCKET"
```

Replace the example values with the actual service-account JSON fields and storage bucket name.

### Login Configuration

The application expects one administrator account and two regular user accounts to be configured.

```toml
[login]
admin_username = "your_admin_username"
admin_password = "your_admin_password"

user1_username = "your_first_user_username"
user1_password = "your_first_user_password"

user2_username = "your_second_user_username"
user2_password = "your_second_user_password"
```

Replace all example usernames and passwords with your own credentials.

**Security requirements**

- Never upload service-account JSON files or `secrets.toml` to GitHub.
- Use strong, unique passwords.
- Restrict access to the Firebase service account.
- Configure the actual storage bucket for your Firebase project.
- The application contains a hardcoded fallback bucket name for local initialization; verify it matches your project if using that fallback.
- The current login implementation compares credentials configured in Streamlit secrets. Use a stronger authentication system before deploying the application for production use.

For Streamlit Community Cloud, add the required secrets through the application's deployment settings.

---

## ▶️ Running the Application

After installing the dependencies and configuring Firebase and login credentials, run:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, typically:

```text
http://localhost:8501
```

Log in using the configured administrator or user credentials.

---

## 👨‍💼 Using the Application

### Administrator Workflow

1. Select **Admin Login**.
2. Enter the configured administrator credentials.
3. Open **Admin Panel: Add Speaker Data**.
4. Enter a speaker's name and metadata.
5. Record five separate voice samples, approximately five seconds each.
6. Upload the samples and save the metadata.
7. Allow the application to train and evaluate the updated model.
8. Use **View/Update Actor Details** to edit saved metadata when needed.

### User Workflow

1. Select **User Login**.
2. Enter the configured user credentials.
3. Choose **Recognize Speaker from File** or **Recognize Speaker Live**.
4. Upload a WAV file or record a voice sample.
5. Review the predicted speaker and any available actor metadata.

Recognition requires a trained model and a dataset containing the enrolled speakers.

---

## 📊 Model Evaluation

The application evaluates the trained model on a held-out test set.

### Metrics and Visualizations

**Accuracy**

Measures the proportion of test samples classified correctly.

\[
\text{Accuracy} =
\frac{\text{Correct Predictions}}{\text{Total Predictions}}
\]

**Classification Report**

Displays precision, recall, F1-score, and support for each speaker class.

**Confusion Matrix**

Visualizes actual versus predicted speaker labels to help identify which speakers the model confuses.

The application does not establish a guaranteed recognition accuracy. Performance depends on dataset size, recording quality, speaker diversity, and the similarity of voices in the enrolled dataset.

---

## ☁️ Firebase Data Organization

### Cloud Storage

The application uses the following storage organization:

```text
Firebase Storage/
│
├── data/
│   ├── Speaker_Name_1/
│   │   ├── sample_1.wav
│   │   ├── sample_2.wav
│   │   └── ...
│   │
│   └── Speaker_Name_2/
│       ├── sample_1.wav
│       └── ...
│
├── speaker_recognition_model.pkl
└── id_to_label_map.pkl
```

The application loads WAV recordings from the `data/` prefix and stores the trained model and label mapping at the bucket's root.

### Cloud Firestore

Speaker metadata is stored in the `actors_actresses_metadata` collection.

Each document uses the speaker's name as its document ID and may contain:

| Field | Description |
|---|---|
| `age` | Actor's or actress's age |
| `height` | Recorded height |
| `total_films` | Total number of films |
| `hit_films` | Number of hit films |
| `industry` | Film industry |
| `last_updated` | Firestore server timestamp |

---

## ⚠️ Limitations

- The model identifies speakers enrolled in its training dataset; it is not a general speech-to-text system.
- Recognition is limited by the available voice samples and their quality.
- The application uses a relatively simple 13-dimensional MFCC representation.
- The current pipeline does not implement a dedicated unknown-speaker rejection threshold.
- A predicted speaker label does not guarantee that the speaker is correctly identified.
- The model requires retraining when new speakers are added.
- The current implementation supports WAV uploads rather than arbitrary audio formats.
- The model's test-set accuracy may not reflect performance on recordings from entirely new environments.
- The application stores serialized model artifacts using pickle; only trusted model files should be loaded.

---

## 🔮 Future Improvements

Potential enhancements include:

- Speaker embeddings using modern deep learning architectures.
- More robust voice activity detection and noise reduction.
- Improved handling of background noise and varying microphones.
- Confidence calibration and unknown-speaker rejection.
- Larger and more balanced training datasets.
- Cross-validation and hyperparameter optimization.
- More detailed recognition history and analytics.
- Stronger authentication and role management.
- Automated tests for audio processing and model prediction.
- Improved model versioning and deployment workflows.

---

## 🎓 Learning Outcomes

This project demonstrates practical experience with:

- Audio signal processing
- MFCC-based feature engineering
- Supervised machine learning
- Random Forest classification
- Model evaluation and confusion matrices
- Python application development
- Streamlit interface development
- Firebase Cloud Storage and Firestore
- Real-time audio capture
- Model serialization and deployment

---

## 👨‍💻 Author

**Yuvansh Kalra**

- GitHub: [@yuvanshkalra](https://github.com/yuvanshkalra)
- Repository: [Voice-recogniser](https://github.com/yuvanshkalra/Voice-recogniser)

---

## ⚖️ Disclaimer

This project is intended for educational and experimental speaker identification. Predictions may be inaccurate, and the application should not be used as the sole basis for identity verification, authentication, or other high-stakes decisions.

Voice recordings are biometric information. Obtain appropriate consent before collecting or