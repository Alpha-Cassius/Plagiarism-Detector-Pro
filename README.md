# 📜 Plagiarism Checker & CSV Model Trainer

This project consists of two applications:
1. **Plagiarism Checker** 🕵️‍♂️: A GUI tool to detect plagiarism in text using a trained machine learning model.
2. **CSV Model Trainer** 📊: A GUI tool to load, edit, and train a plagiarism detection model using CSV data.

---

## 🚀 Features
### Plagiarism Checker
- 🔍 Detects plagiarism using a trained ML model.
- 📄 Allows text input for analysis.
- ✅ Provides clear feedback on plagiarism detection.
- 🎨 User-friendly interface with a modern dark theme.

### CSV Model Trainer
- 📂 Load and process CSV files with text data.
- ✏️ Edit dataset entries (add/delete rows).
- 🏋️ Train a model using **Logistic Regression, Random Forest, or SVM**.
- 💾 Save the trained model for use in the Plagiarism Checker.

---

## 🛠 Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Alpha-Cassius/Plagiarism-Detector-Pro.git
   cd Plagiarism-Detector-Pro
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the applications**:
   - Plagiarism Checker:
     ```bash
     python app.py
     ```
   - CSV Model Trainer:
     ```bash
     python train.py
     ```

---

## 🏗 How It Works
### Plagiarism Detection Flow
1. User enters text.
2. The text is vectorized using **TF-IDF**.
3. The trained model predicts whether the text is plagiarized or not.
4. The result is displayed with color-coded feedback.

### Training Model Flow
1. Load a CSV file containing `source_text`, `plagiarized_text`, and `label`.
2. Select a machine learning algorithm (**Logistic Regression, Random Forest, SVM**).
3. Train the model and evaluate accuracy.
4. Save the trained model for use in the Plagiarism Checker.

---

## 📂 File Structure
```
📁 plagiarism-checker/
│── app.py              # Plagiarism Checker GUI
│── train.py            # CSV Model Trainer GUI
│── model.pkl           # Saved ML model (generated after training)
│── vectorizer.pkl      # TF-IDF vectorizer (generated after training)
│── requirements.txt    # Required dependencies
```

---

## 📜 License
This project is open-source and available for personal and educational use.

---

## 👨‍💻 Author
Developed by **Alpha Cassius**. 🚀

