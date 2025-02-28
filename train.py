import customtkinter as ctk
import pandas as pd
import pickle
from tkinter import filedialog, messagebox, ttk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Initialize GUI
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")
root = ctk.CTk()
root.title("CSV Model Trainer")
root.geometry("850x700")
root.resizable(False, False)

# Global Variables
df = pd.DataFrame(columns=["Index", "source_text", "plagiarized_text", "label"])
tfidf_vectorizer = TfidfVectorizer()
model = None

# Function to load CSV
def load_csv():
    global df
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        df = pd.read_csv(file_path)

        if "Index" not in df.columns:
            df.insert(0, "Index", range(len(df)))  # Assign auto-index if missing

        update_table()
        messagebox.showinfo("Success", "CSV loaded successfully!")

# Function to update table
def update_table():
    tree.delete(*tree.get_children())  
    df.reset_index(drop=True, inplace=True)  
    df["Index"] = df.index  

    for _, row in df.iterrows():
        tree.insert("", "end", values=(row["Index"], row["source_text"], row["plagiarized_text"], row["label"]))

# Function to add row
def add_row():
    global df
    try:
        if not entry_source.get().strip() or not entry_plag.get().strip() or not entry_label.get().strip():
            raise ValueError("All fields are required!")
        
        new_row = pd.DataFrame([{
            "source_text": entry_source.get().strip(),
            "plagiarized_text": entry_plag.get().strip(),
            "label": int(entry_label.get().strip())
        }])
        
        df = pd.concat([df, new_row], ignore_index=True)  
        df["Index"] = df.index  
        update_table()  
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")

# Function to delete row
def delete_row():
    global df
    try:
        row_idx = int(entry_index.get().strip())
        if row_idx not in df["Index"].values:
            raise ValueError("Index not found")

        df = df[df["Index"] != row_idx]  
        df.reset_index(drop=True, inplace=True)  
        df["Index"] = df.index  
        update_table()
    except Exception as e:
        messagebox.showerror("Error", f"Invalid row index: {e}")

# Function to train model
def train_model():
    global model, tfidf_vectorizer
    if df.empty:
        messagebox.showerror("Error", "No data available")
        return

    # Split dataset
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["source_text"] + " " + df["plagiarized_text"], df["label"], test_size=0.2, random_state=42
    )

    # Fit vectorizer only on training data
    X_train = tfidf_vectorizer.fit_transform(X_train_text)
    X_test = tfidf_vectorizer.transform(X_test_text)  # Use transform instead of fit_transform

    # Choose model
    model_type = model_var.get()
    if model_type == "Logistic Regression":
        model = LogisticRegression()
    elif model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = SVC(kernel='linear', random_state=42)

    # Train model
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    messagebox.showinfo("Model Trained", f"Accuracy: {accuracy:.4f}\n{classification_report(y_test, y_pred)}")


# Function to save model
def save_model():
    if model:
        pickle.dump(model, open("model.pkl", "wb"))
        pickle.dump(tfidf_vectorizer, open("vectorizer.pkl", "wb"))
        messagebox.showinfo("Success", "Model saved successfully!")
    else:
        messagebox.showerror("Error", "No trained model to save")

# UI Components
frame = ctk.CTkFrame(root, width=800, height=50)
frame.place(x=25, y=10)

btn_load = ctk.CTkButton(frame, text="Load CSV", command=load_csv)
btn_load.place(x=10, y=10)

btn_train = ctk.CTkButton(frame, text="Train Model", command=train_model)
btn_train.place(x=170, y=10)

btn_save = ctk.CTkButton(frame, text="Save Model", command=save_model)
btn_save.place(x=330, y=10)

model_var = ctk.StringVar(value="Logistic Regression")
model_dropdown = ctk.CTkComboBox(frame, values=["Logistic Regression", "Random Forest", "SVM"], variable=model_var)
model_dropdown.place(x=490, y=10)

# Table Frame
tree_frame = ctk.CTkFrame(root, width=800, height=400)  
tree_frame.place(x=25, y=70)

tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical")
tree_scroll.pack(side="right", fill="y")

tree = ttk.Treeview(tree_frame, columns=("Index", "Source Text", "Plagiarized Text", "Label"), show="headings", yscrollcommand=tree_scroll.set)

tree.heading("Index", text="Index")
tree.heading("Source Text", text="Source Text")
tree.heading("Plagiarized Text", text="Plagiarized Text")
tree.heading("Label", text="Label")

# Adjust column widths
tree.column("Index", width=50, anchor="center")
tree.column("Source Text", width=250, anchor="w")
tree.column("Plagiarized Text", width=250, anchor="w")
tree.column("Label", width=50, anchor="center")

tree.pack(expand=True, fill="both")
tree_scroll.config(command=tree.yview)

# Input Fields
entry_source = ctk.CTkEntry(root, placeholder_text="Source Text", width=400)
entry_source.place(x=25, y=480)
entry_plag = ctk.CTkEntry(root, placeholder_text="Plagiarized Text", width=400)
entry_plag.place(x=25, y=520)
entry_label = ctk.CTkEntry(root, placeholder_text="Label (0 or 1)", width=170)
entry_label.place(x=25, y=560)

btn_add = ctk.CTkButton(root, text="Add Row", command=add_row)
btn_add.place(x=240, y=560)

entry_index = ctk.CTkEntry(root, placeholder_text="Row Index to Delete", width=150)
entry_index.place(x=670, y=480)
btn_delete = ctk.CTkButton(root, text="Delete Row", command=delete_row)
btn_delete.place(x=670, y=520)

root.mainloop()
