
# 🎙️ Voice Recogniser — ML-Based Speaker Identification

An interactive speaker-recognition application built with Python, Streamlit, and machine learning. The application captures voice recordings, extracts meaningful audio features, and uses a Random Forest classifier to identify speakers from their voice samples.

## 🚀 Overview

Voice Recogniser is a machine-learning project designed to explore speaker identification through audio signal processing and supervised learning.

The application combines audio recording, feature extraction, model training, speaker-label mapping, and Firebase integration in a web-based interface.

### ✨ Key Features

- **Voice Recording:** Capture audio samples through an interactive Streamlit interface.
- **Audio Feature Extraction:** Extract Mel-Frequency Cepstral Coefficients (MFCCs) using Librosa.
- **Machine Learning:** Use a Random Forest classifier for speaker identification.
- **Model Evaluation:** Evaluate classification performance using accuracy, classification reports, and confusion matrices.
- **Firebase Integration:** Integrate Firebase Storage and Cloud Firestore for audio and metadata management.
- **Data Management:** Use Pandas to manage and organize speaker-related information.
- **Visual Analytics:** Generate visualizations to inspect model performance.
- **Interactive UI:** Access the application's functionality through a browser-based Streamlit interface.

## 🧠 How It Works

The application follows a speaker-recognition pipeline:

1. **Audio Input:** Record voice samples using the application's audio recording interface.
2. **Preprocessing:** Load and process audio signals into a suitable format.
3. **Feature Extraction:** Extract MFCC features using Librosa to represent important characteristics of the voice.
4. **Model Training:** Train a Random Forest classifier on labelled speaker samples.
5. **Prediction:** Use the trained model to classify a voice recording.
6. **Evaluation:** Assess model performance using classification metrics and a confusion matrix.
7. **Storage:** Integrate Firebase services for audio storage and speaker metadata management.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application and ML logic |
| Streamlit | Interactive web interface |
| NumPy | Numerical computation |
| SciPy | Audio signal processing |
| Librosa | Audio analysis and MFCC extraction |
| Scikit-learn | Random Forest classification and evaluation |
| Pandas | Data manipulation |
| Matplotlib | Data visualization |
| Seaborn | Confusion matrix visualization |
| Firebase Admin SDK | Firebase Storage and Firestore integration |
| streamlit-audiorec | Audio recording interface |

## 📁 Project Structure

```text
Voice-recogniser/
│
├── app.py            # Main Streamlit application
├── requirements.txt  # Python dependencies
├── README.md         # Project documentation
└── sso_logo.png      # Application logo or visual asset
```

The application may also create temporary recordings and model artifacts during execution.

## ⚙️ Installation and Setup

### Prerequisites

- Python installed on your system
- Git
- A working microphone
- A Firebase project if you want to use cloud storage and metadata features

### 1. Clone the Repository

```bash
git clone https://github.com/yuvanshkalra/Voice-recogniser.git
cd Voice-recogniser
```

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 🔥 Firebase Configuration

The application uses the Firebase Admin SDK to connect to Firebase services.

To configure your own Firebase project:

1. Create a project in the [Firebase Console](https://console.firebase.google.com/).
2. Enable Cloud Firestore and Firebase Storage as required.
3. Generate a service account credential securely from your Firebase project settings.
4. Configure the service account JSON and storage bucket for the application.
5. Set the required values in Streamlit secrets.

Create `.streamlit/secrets.toml`:

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

Replace the placeholder values with your actual service account details. Use the complete JSON credential provided by Firebase, including any additional fields it contains.

**Security:** Never commit Firebase credentials, private keys, or populated secrets files to GitHub. Keep them in your local secrets configuration or your deployment platform's secret manager.

The application also contains a local credential-file fallback. If you use that approach, configure the expected `firebase_service_account.json` file securely and ensure it is excluded from version control.

## ▶️ Run the Application

Start the Streamlit application from the project directory:

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to access the application.

## 📊 Machine Learning Approach

### Feature Extraction

The project uses Mel-Frequency Cepstral Coefficients (MFCCs) to represent characteristics of the recorded audio.

MFCCs are widely used in speech and speaker-recognition systems because they capture information about the spectral envelope of an audio signal.

### Classification Algorithm

The application uses a Random Forest classifier from Scikit-learn.

Random Forest combines multiple decision trees to produce a classification prediction. Its performance depends on the quality of the audio features, training samples, and speaker labels.

### Model Evaluation

The code includes tools for evaluating the classifier:

- Accuracy score
- Classification report
- Confusion matrix

Actual accuracy and other performance metrics should be measured on held-out recordings. No performance figure is claimed here.

## 🎯 Applications

- Speaker identification experiments
- Voice-based ML demonstrations
- Audio signal processing projects
- Educational demonstrations of supervised learning
- Prototyping voice-enabled applications

## ⚠️ Limitations

- Recognition performance depends on recording quality, background noise, and the available training samples.
- A speaker may be misclassified when their recording conditions differ significantly from the training data.
- The application requires suitable audio-recording support in the environment where it runs.
- Firebase-dependent features require valid credentials and correctly configured cloud resources.
- Speaker identification is not the same as speech-to-text transcription or voice authentication.

## 🔮 Future Improvements

- Add a dedicated interface for uploading audio files.
- Improve robustness against background noise.
- Evaluate performance using a larger and more diverse speaker dataset.
- Add cross-validation and more detailed model evaluation.
- Compare Random Forest with alternative speaker-recognition algorithms.
- Add automated tests and continuous integration.
- Improve deployment configuration and secret management.

## 👨‍💻 Author

**Yuvansh Kalra**

GitHub: [@yuvanshkalra](https://github.com/yuvanshkalra)

Project Repository: [Voice-recogniser](https://github.com/yuvanshkalra/Voice-recogniser)

## 📄 License

No license is specified in this repository. Contact the repository owner for permission before redistributing or reusing the project.

---

⭐ If you find this project useful, consider starring the repository.
