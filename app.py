import customtkinter as ctk
import pickle

# Load the model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

def detect(input_text):
    vectorized_text = tfidf_vectorizer.transform([input_text])
    result = model.predict(vectorized_text)
    return ["✅ No Plagiarism Detected", "light green"] if result[0] == 0 else ["❌ Plagiarism Detected", "red"]

# Configure UI theme
ctk.set_appearance_mode("dark")  
ctk.set_default_color_theme("blue")  

# Main Application Window
app = ctk.CTk()
app.title("Plagiarism Checker")
app.geometry("600x500")
app.resizable(False, False)

# Heading
title_label = ctk.CTkLabel(app, text="Plagiarism Checker", font=("Arial", 24, "bold"))
title_label.pack(pady=15)

# Main Frame
frame = ctk.CTkFrame(app, width=550, height=350, corner_radius=10)
frame.pack(pady=10, padx=20, fill="both", expand=True)

# Text Entry
text_entry = ctk.CTkTextbox(frame, width=500, height=150, wrap="word", font=("Arial", 14))
text_entry.pack(pady=15, padx=10)

# Function to handle button click
def check_plagiarism():
    input_text = text_entry.get("1.0", "end-1c").strip()
    if not input_text:
        result_label.configure(text="⚠️ Please enter text!", text_color="yellow", width=400)
        return
    detection_result = detect(input_text)
    result_label.configure(text=detection_result[0], text_color=detection_result[1], width=400)

# Button
check_button = ctk.CTkButton(frame, text="Check Plagiarism", command=check_plagiarism, font=("Arial", 16))
check_button.pack(pady=10)

# Result Label with Fixed Width
result_label = ctk.CTkLabel(frame, text="", font=("Arial", 16, "bold"), width=400, anchor="center")
result_label.pack(pady=10)

# Footer
footer_label = ctk.CTkLabel(app, text="Created by Alpha Cassius", font=("Arial", 12))
footer_label.pack(pady=10)

# Run the app
app.mainloop()
