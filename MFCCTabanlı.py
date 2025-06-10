# === Imports ===
import os
import time 
import random
import joblib
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.models import Sequential, load_model # type: ignore
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization # type: ignore

# === Dataset loader ===
def load_dataset_labels(folder_path):
    data = []
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            try:
                name, emotion, index = file[:-4].split("_")
                data.append({
                    "filename": file,
                    "name": name,
                    "emotion": emotion,
                    "sentence_number": int(index)
                })
            except ValueError:
                print(f"Invalid format: {file}")
    return pd.DataFrame(data)

# === Train/test split ===
def group_split(folder_path, test_ratio=0.2):
    df = load_dataset_labels(folder_path)
    train_list, test_list = [], []

    grouped = df.groupby(['name', 'emotion'])
    for (name, emotion), group in grouped:
        filenames = group['filename'].tolist()
        train_files, test_files = train_test_split(filenames, test_size=test_ratio)
        train_list.extend(train_files)
        test_list.extend(test_files)

    return train_list, test_list

# === Confusion Matrices ===
def plot_confusions(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Confusion Matrix Analysis", fontsize=16)

    name_classes = sorted(list(set(df["true_name"]) | set(df["predicted_name"])))
    cm_name = confusion_matrix(df["true_name"], df["predicted_name"], labels=name_classes)
    disp_name = ConfusionMatrixDisplay(confusion_matrix=cm_name, display_labels=name_classes)
    disp_name.plot(ax=axes[0], cmap="Greens", values_format='d', colorbar=False)
    axes[0].set_title("Name Confusion Matrix", pad=10)
    axes[0].tick_params(axis='x', rotation=45)

    emotion_classes = sorted(list(set(df["true_emotion"]) | set(df["predicted_emotion"])))
    cm_emotion = confusion_matrix(df["true_emotion"], df["predicted_emotion"], labels=emotion_classes)
    disp_emotion = ConfusionMatrixDisplay(confusion_matrix=cm_emotion, display_labels=emotion_classes)
    disp_emotion.plot(ax=axes[1], cmap="Greens", values_format='d', colorbar=False)
    axes[1].set_title("Emotion Confusion Matrix", pad=10)
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

# === Score tables ===
def plot_scores(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 0.5 * len(df["true_name"].unique()) + 4))
    fig.suptitle("Recognition Score Tables", fontsize=16)

    # === NAME SCORES ===
    name_scores = score_table(df, label_type="name")
    axes[0].axis('off')

    font_size = max(6, 12 - len(name_scores) // 3)
    scale_y = 1.0 + len(name_scores) * 0.05

    table1 = axes[0].table(
        cellText=name_scores.values,
        colLabels=name_scores.columns,
        loc='center',
        cellLoc='center',
        colColours=["#7bffa1"] * len(name_scores.columns),
        cellColours=[["#ffffff"] * len(name_scores.columns) for _ in range(len(name_scores))],
        edges='closed'
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(font_size)
    table1.scale(1, scale_y)
    axes[0].set_title("Name Scores", pad=6)

    # === EMOTION SCORES ===
    emotion_scores = score_table(df, label_type="emotion")
    axes[1].axis('off')

    font_size2 = max(6, 12 - len(emotion_scores) // 3)
    scale_y2 = 1.0 + len(emotion_scores) * 0.05

    table2 = axes[1].table(
        cellText=emotion_scores.values,
        colLabels=emotion_scores.columns,
        loc='center',
        cellLoc='center',
        colColours=["#7bffa1"] * len(emotion_scores.columns),
        cellColours=[["#ffffff"] * len(emotion_scores.columns) for _ in range(len(emotion_scores))],
        edges='closed'
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(font_size2)
    table2.scale(1, scale_y2)
    axes[1].set_title("Emotion Scores", pad=6)

    plt.tight_layout(rect=[0, 0, 1, 0.9])

# === Score calculator ===
def score_table(df, label_type="name"):
    if label_type == "name":
        true_col = "true_name"
        pred_col = "predicted_name"
    elif label_type == "emotion":
        true_col = "true_emotion"
        pred_col = "predicted_emotion"
    else:
        raise ValueError("label_type must be 'name' or 'emotion'")

    classes = sorted(df[true_col].unique())
    results = []

    for cls in classes:
        y_true = df[true_col].apply(lambda x: 1 if x == cls else 0)
        y_pred = df[pred_col].apply(lambda x: 1 if x == cls else 0)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        results.append({
            label_type: cls,
            "Accuracy": round(acc, 3),
            "Precision": round(prec, 3),
            "Recall": round(rec, 3),
            "F1": round(f1, 3)
        })

    avg_f1 = sum([r["F1"] for r in results]) / len(results)
    results.append({
        label_type: "AVERAGE",
        "Accuracy": "-",
        "Precision": "-",
        "Recall": "-",
        "F1": round(avg_f1, 3)
    })
    return pd.DataFrame(results)

# === MFCC Extraction ===
def extract_mfcc(file_path, target_duration=4.0, sr=16000, n_mfcc=13, hop_length=160, n_fft=400):
    y, _ = librosa.load(file_path, sr=sr)
    target_length = int(target_duration * sr)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), mode='constant')
    else:
        y = y[:target_length]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    mfcc = mfcc.reshape(13, 401, 1)
    return mfcc

# === Input File List (Prediction Mode) ===
def list_wav_files(folder="Input"):
    return [f for f in os.listdir(folder) if f.lower().endswith(".wav")]

# === CNN Model Training & Inference ===
def run_model(wav_list, mode="Pred", model_path_emotion="trained_model_emotion.keras", model_path_name="trained_model_name.keras"):

    if mode == "train":
        start_time = time.time()
        X, y_emotion, y_name = [], [], []

        for filename in wav_list:
            path = os.path.join("Dataset", filename)
            if not os.path.exists(path):
                print(f"File not found: {path}")
                continue

            mfcc = extract_mfcc(path)
            X.append(mfcc)

            try:
                name, emotion, _ = filename[:-4].split("_")
            except:
                print(f"Invalid filename format: {filename}")
                continue

            y_emotion.append(emotion)
            y_name.append(name)

        X = np.array(X)

        early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

        # Emotion model
        le_emotion = LabelEncoder()
        y_emotion_enc = le_emotion.fit_transform(y_emotion)
        y_emotion_cat = to_categorical(y_emotion_enc)

        model_emotion = Sequential([
            Input(shape=(13, 401, 1)),
            Conv2D(32, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(y_emotion_cat.shape[1], activation='softmax')
        ])

        model_emotion.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model_emotion.fit(X, y_emotion_cat, epochs=200, batch_size=32, callbacks=[early_stop])
        model_emotion.save(model_path_emotion)
        joblib.dump(le_emotion, "label_encoder_emotion.pkl")

        # Name model
        le_name = LabelEncoder()
        y_name_enc = le_name.fit_transform(y_name)
        y_name_cat = to_categorical(y_name_enc)

        model_name = Sequential([
            Input(shape=(13, 401, 1)),
            Conv2D(32, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(y_name_cat.shape[1], activation='softmax')
        ])

        model_name.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model_name.fit(X, y_name_cat, epochs=200, batch_size=32, callbacks=[early_stop])
        model_name.save(model_path_name)
        joblib.dump(le_name, "label_encoder_name.pkl")

        print("CNN models for name and emotion recognition saved.")

        end_time = time.time()
        print(f"[TRAIN] time: {end_time - start_time:.2f} sec")
    elif mode == "Pred":
        start_time = time.time()
        model_emotion = load_model(model_path_emotion)
        model_name = load_model(model_path_name)
        le_emotion = joblib.load("label_encoder_emotion.pkl")
        le_name = joblib.load("label_encoder_name.pkl")

        results = []
        for filename in wav_list:
            path = os.path.join("Input", filename)
            if not os.path.exists(path):
                print(f"File not found: {path}")
                continue

            mfcc = extract_mfcc(path)
            mfcc = np.asarray(mfcc, dtype=np.float32).reshape(1, 13, 401, 1)

            pred_emotion = model_emotion.predict(mfcc)
            pred_name = model_name.predict(mfcc)

            pred_emotion_label = le_emotion.inverse_transform([np.argmax(pred_emotion)])[0]
            pred_name_label = le_name.inverse_transform([np.argmax(pred_name)])[0]

            results.append({
                "filename": filename,
                "predicted_emotion": pred_emotion_label,
                "predicted_name": pred_name_label
            })
        end_time = time.time()
        print(f"[PRED] time: {end_time - start_time:.2f} sec")
        return pd.DataFrame(results)
    elif mode == "test":
        start_time = time.time()
        model_emotion = load_model(model_path_emotion)
        model_name = load_model(model_path_name)
        le_emotion = joblib.load("label_encoder_emotion.pkl")
        le_name = joblib.load("label_encoder_name.pkl")

        results = []
        for filename in wav_list:
            path = os.path.join("Dataset", filename)
            if not os.path.exists(path):
                print(f"File not found: {path}")
                continue

            mfcc = extract_mfcc(path)

            try:
                true_name, true_emotion, _ = filename[:-4].split("_")
            except:
                print(f"Invalid filename format: {filename}")
                continue

            pred_emotion = model_emotion.predict(np.expand_dims(mfcc, axis=0))
            pred_name = model_name.predict(np.expand_dims(mfcc, axis=0))

            pred_emotion_label = le_emotion.inverse_transform([np.argmax(pred_emotion)])[0]
            pred_name_label = le_name.inverse_transform([np.argmax(pred_name)])[0]

            results.append({
                "filename": filename,
                "true_name": true_name,
                "predicted_name": pred_name_label,
                "true_emotion": true_emotion,
                "predicted_emotion": pred_emotion_label
            })
        end_time = time.time()
        print(f"[TEST] time: {end_time - start_time:.2f} sec")
        return pd.DataFrame(results)
    else:
        raise ValueError("Invalid mode: choose from 'train', 'Pred', or 'test'")

# === Repeated Model Evaluation ===
def evaluate_model_stats(folder_path="Dataset", repeat=10,split_ratio = 0.2):
    f1_name_scores = []
    f1_emotion_scores = []

    for i in range(repeat):
        os.system("cls" if os.name == "nt" else "clear")
        print(f"\nIteration {i+1} started...")
        train_list, test_list = group_split(folder_path,test_ratio=split_ratio)

        run_model(train_list, mode="train")
        df_result = run_model(test_list, mode="test")

        name_f1 = f1_score(df_result["true_name"], df_result["predicted_name"], average="macro", zero_division=0)
        emotion_f1 = f1_score(df_result["true_emotion"], df_result["predicted_emotion"], average="macro", zero_division=0)

        f1_name_scores.append(name_f1)
        f1_emotion_scores.append(emotion_f1)

    avg_name = np.mean(f1_name_scores)
    avg_emotion = np.mean(f1_emotion_scores)
    std_name = np.std(f1_name_scores)
    std_emotion = np.std(f1_emotion_scores)
    var_name = np.var(f1_name_scores)
    var_emotion = np.var(f1_emotion_scores)

    # 95% Confidence Interval (z = 1.96)
    ci_name = 1.96 * std_name / np.sqrt(repeat)
    ci_emotion = 1.96 * std_emotion / np.sqrt(repeat)

    print("\n=== Average F1 Scores ===")
    print(f"Name Recognition Avg F1: {avg_name:.3f} ± {std_name:.3f} (Var: {var_name:.5f}, CI95: ±{ci_name:.3f})")
    print(f"Emotion Recognition Avg F1: {avg_emotion:.3f} ± {std_emotion:.3f} (Var: {var_emotion:.5f}, CI95: ±{ci_emotion:.3f})")

    return avg_name, avg_emotion, f1_name_scores, f1_emotion_scores

# === Iterative F1 Score Plot ===
def plot_analysis(max_iter=10, folder_path="Dataset", split_ratio = 0.2):
    avg_name, avg_emotion, f1_name_scores, f1_emotion_scores = evaluate_model_stats(folder_path, repeat=max_iter,split_ratio=split_ratio)

    x_vals = list(range(1, max_iter + 1))
    y_vals_name = [np.mean(f1_name_scores[:i]) for i in x_vals]
    y_vals_emotion = [np.mean(f1_emotion_scores[:i]) for i in x_vals]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Cumulative Average F1 Scores and Distributions")

    # Emotion Line Plot
    axes[0, 0].plot(x_vals, y_vals_emotion, marker='o', color='orange')
    axes[0, 0].set_title("Emotion Recognition Avg F1")
    axes[0, 0].set_xlabel("Iteration Count")
    axes[0, 0].set_ylabel("F1 Score")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(True)
    axes[0, 0].set_xticks(x_vals)

    # Name Line Plot
    axes[0, 1].plot(x_vals, y_vals_name, marker='o', color='blue')
    axes[0, 1].set_title("Name Recognition Avg F1")
    axes[0, 1].set_xlabel("Iteration Count")
    axes[0, 1].set_ylabel("F1 Score")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(True)
    axes[0, 1].set_xticks(x_vals)

    # Emotion Boxplot
    axes[1, 0].boxplot(f1_emotion_scores, vert=True, patch_artist=True, showmeans=True, tick_labels=['Emotion'])
    axes[1, 0].set_title("Emotion F1 Score Distribution")
    axes[1, 0].set_ylabel("F1 Score")
    axes[1, 0].set_ylim(0, 1)

    # Name Boxplot
    axes[1, 1].boxplot(f1_name_scores, vert=True, patch_artist=True, showmeans=True, tick_labels=['Name'])
    axes[1, 1].set_title("Name F1 Score Distribution")
    axes[1, 1].set_ylabel("F1 Score")
    axes[1, 1].set_ylim(0, 1)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


#=====================================================================
train_list, test_list = group_split("Dataset", test_ratio=0.375)
#=====================================================================
run_model(train_list, mode="train")
#=====================================================================
test_Result = run_model(test_list, mode="test")
#=====================================================================
wav_Input_list = list_wav_files()
df_pred = run_model(wav_Input_list, mode="Pred")
print(df_pred)
#=====================================================================
plot_confusions(test_Result)
plot_scores(test_Result)
plt.show()
#=====================================================================
'''
plot_analysis(max_iter=20,split_ratio=0.375)
'''