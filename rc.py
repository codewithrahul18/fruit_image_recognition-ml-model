import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="NeuroFruit AI Pro", layout="wide", page_icon="🍎")

# --- CSS STYLING ---
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 10px; font-weight: bold; }
    .reportview-container { background: #f0f2f6; }
    .metric-card { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'model' not in st.session_state: st.session_state.model = None
if 'class_names' not in st.session_state: st.session_state.class_names = []
if 'history' not in st.session_state: st.session_state.history = None
if 'trained' not in st.session_state: st.session_state.trained = False

# --- MODEL ARCHITECTURE ---
def create_model(num_classes, fine_tune=False, learning_rate=0.001):
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    
    if fine_tune:
        base_model.trainable = True
        # Fine-tune the last 40 layers for high accuracy
        for layer in base_model.layers[:-40]:
            layer.trainable = False
        lr = learning_rate / 10  # Lower LR for fine-tuning
    else:
        base_model.trainable = False
        lr = learning_rate

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x) # Increased capacity
    x = Dropout(0.4)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- TRAINING FUNCTION ---
def train_pipeline(dataset_path, epochs, batch_size, fine_tune):
    # Validate Path
    if not os.path.exists(dataset_path):
        return None, None, None, "Path not found!"
    
    classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    if not classes:
        return None, None, None, "No class folders found!"

    # Data Augmentation (Aggressive for better generalization)
    datagen = ImageDataGenerator(
        rescale=1./255, 
        rotation_range=30, 
        width_shift_range=0.2, 
        height_shift_range=0.2, 
        shear_range=0.2, 
        zoom_range=0.2, 
        horizontal_flip=True, 
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        dataset_path, target_size=(224, 224), batch_size=batch_size, class_mode='categorical', subset='training'
    )
    val_gen = datagen.flow_from_directory(
        dataset_path, target_size=(224, 224), batch_size=batch_size, class_mode='categorical', subset='validation', shuffle=False
    )

    model = create_model(len(classes), fine_tune=fine_tune)
    
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-6)
    ]

    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)
    return model, classes, history, "Success", val_gen

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3075/3075977.png", width=80)
    st.title("NeuroFruit Settings")
    
    st.subheader("📁 Data Configuration")
    dataset_path = st.text_input("Dataset Path", value="dataset", help="Path to folder containing fruit subfolders")
    
    st.subheader("⚙️ Hyperparameters")
    epochs = st.slider("Epochs", 5, 100, 20)
    batch_size = st.select_slider("Batch Size", options=[8, 16, 32, 64], value=16)
    fine_tune = st.checkbox("Enable High Accuracy (Fine-Tuning)", value=True, help="Unfreezes layers to reach 99% accuracy. Slower but better.")
    
    st.markdown("---")
    st.subheader("🎮 Controls")
    
    # THE 3 BUTTONS
    btn_train = st.button("🚀 Train Model", type="primary")
    btn_report = st.button("📊 Generate Report")
    btn_reset = st.button("🔄 Reset System")

# --- MAIN LOGIC ---
if btn_reset:
    st.session_state.clear()
    st.rerun()

st.title("🍎 NeuroFruit AI: High-Precision Recognition")
st.markdown("Using **MobileNetV2** with Transfer Learning & Fine-Tuning capabilities.")

# TABS
tab_train, tab_predict, tab_viz = st.tabs(["🧠 Training Dashboard", "🔍 Live Prediction", "📈 Analytics"])

# --- TAB 1: TRAINING ---
with tab_train:
    if btn_train:
        with st.status("Training Neural Network...", expanded=True) as status:
            st.write("Initializing MobileNetV2 base...")
            st.write("Augmenting data images...")
            model, classes, history, msg, val_gen = train_pipeline(dataset_path, epochs, batch_size, fine_tune)
            
            if model:
                st.session_state.model = model
                st.session_state.class_names = classes
                st.session_state.history = history
                st.session_state.trained = True
                st.session_state.val_gen = val_gen # Save for reporting
                status.update(label="Training Complete!", state="complete", expanded=False)
                st.success(f"Model Trained Successfully on {len(classes)} classes!")
            else:
                st.error(f"Error: {msg}")

    if st.session_state.trained and st.session_state.history:
        st.subheader("Training Performance")
        hist = st.session_state.history.history
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("Accuracy Curve")
            fig_acc, ax_acc = plt.subplots()
            ax_acc.plot(hist['accuracy'], label='Train Acc')
            ax_acc.plot(hist['val_accuracy'], label='Val Acc')
            ax_acc.set_title('Model Accuracy')
            ax_acc.legend()
            st.pyplot(fig_acc)
            
        with col2:
            st.info("Loss Curve")
            fig_loss, ax_loss = plt.subplots()
            ax_loss.plot(hist['loss'], label='Train Loss')
            ax_loss.plot(hist['val_loss'], label='Val Loss')
            ax_loss.set_title('Model Loss')
            ax_loss.legend()
            st.pyplot(fig_loss)

# --- TAB 2: PREDICTION ---
with tab_predict:
    col_upload, col_result = st.columns([1, 1.5])
    
    with col_upload:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose a fruit image...", type=['jpg', 'png', 'jpeg'])
        if uploaded_file:
            image = Image.open(uploaded_file)
            # Use updated container width parameter
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
    with col_result:
        st.subheader("AI Diagnosis")
        if uploaded_file and st.session_state.trained:
            if st.button("Analyze Image"):
                # *** CRITICAL FIX: Convert to RGB to ensure 3 channels ***
                img = image.convert('RGB').resize((224, 224))
                
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Predict
                preds = st.session_state.model.predict(img_array)
                score = np.max(preds)
                class_idx = np.argmax(preds)
                fruit_name = st.session_state.class_names[class_idx]
                
                # Display Metrics
                st.metric(label="Detected Fruit", value=fruit_name.upper())
                
                # Accuracy Meter
                st.write("Confidence Score:")
                st.progress(float(score))
                st.caption(f"The model is {score:.2%} sure this is a {fruit_name}.")
                
                # Top 3 Probabilities Chart
                st.write("Top 3 Predictions:")
                top_3_indices = np.argsort(preds[0])[-3:][::-1]
                top_3_values = [preds[0][i] for i in top_3_indices]
                top_3_labels = [st.session_state.class_names[i] for i in top_3_indices]
                
                chart_data = pd.DataFrame({"Fruit": top_3_labels, "Probability": top_3_values})
                st.bar_chart(chart_data, x="Fruit", y="Probability")
                
        elif not st.session_state.trained:
            st.warning("⚠️ Please train the model in the Dashboard tab first.")

# --- TAB 3: REPORTING ---
with tab_viz:
    if btn_report:
        if st.session_state.trained and 'val_gen' in st.session_state:
            st.subheader("Model Evaluation Report")
            with st.spinner("Calculating Confusion Matrix..."):
                val_gen = st.session_state.val_gen
                val_gen.reset() # Reset generator
                
                # Get Predictions for whole validation set
                Y_pred = st.session_state.model.predict(val_gen)
                y_pred = np.argmax(Y_pred, axis=1)
                y_true = val_gen.classes
                
                # Confusion Matrix
                cm = confusion_matrix(y_true, y_pred)
                fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=st.session_state.class_names, 
                            yticklabels=st.session_state.class_names)
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                st.pyplot(fig_cm)
                
                # Text Report
                st.text("Classification Report:")
                report = classification_report(y_true, y_pred, target_names=st.session_state.class_names)
                st.code(report)
        else:
            st.error("Train the model first to generate a report.")
    else:
        st.info("Click 'Generate Report' in the sidebar to see advanced metrics.")


# python -m streamlit run rc.py

