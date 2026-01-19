"""
COMPLETE FRUIT CLASSIFICATION SYSTEM
All features in one file - Training + Prediction + Image Display
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

# Settings
IMG_SIZE = (64, 64)
BATCH_SIZE = 8
EPOCHS = 10
MODEL_NAME = "fruit_model_complete.h5"

class FruitClassifier:
    def __init__(self):
        self.fruits = ['apple', 'banana', 'grapes', 'mango', 'orange', 'pineapple', 'strawberry']
        self.model = None
    
    def train_model(self):
        """Train the fruit classification model"""
        print("TRAINING MODEL...")
        
        # Data loading
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)
        
        train_data = datagen.flow_from_directory(
            directory="dataset",
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            subset="training"
        )
        
        if train_data.samples > 10:
            val_data = datagen.flow_from_directory(
                directory="dataset",
                target_size=IMG_SIZE,
                batch_size=BATCH_SIZE,
                class_mode="categorical",
                subset="validation"
            )
        else:
            val_data = None
        
        # Model architecture
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(64,64,3)),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(len(self.fruits), activation="softmax")
        ])
        
        # Compile and train
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        
        if val_data:
            self.model.fit(train_data, validation_data=val_data, epochs=EPOCHS, verbose=1)
        else:
            self.model.fit(train_data, epochs=EPOCHS, verbose=1)
        
        # Save model
        self.model.save(MODEL_NAME)
        print(f"Model saved as {MODEL_NAME}")
    
    def load_model(self):
        """Load existing model"""
        if os.path.exists(MODEL_NAME):
            self.model = tf.keras.models.load_model(MODEL_NAME)
            print(f"Model loaded from {MODEL_NAME}")
            return True
        return False
    
    def predict_fruit(self, img_path, show_image=True, save_result=False):
        """Predict fruit with image display"""
        if not self.model:
            print("No model loaded!")
            return None
        
        # Process image
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = self.model.predict(img_array, verbose=0)
        predicted_idx = np.argmax(predictions)
        predicted_fruit = self.fruits[predicted_idx]
        confidence = predictions[0][predicted_idx] * 100
        
        # Show result
        print(f"INPUT: {os.path.basename(img_path)}")
        print(f"OUTPUT: {predicted_fruit.upper()} ({confidence:.1f}%)")
        
        # Display image
        if show_image:
            plt.figure(figsize=(8, 6))
            plt.imshow(image.load_img(img_path))
            plt.title(f"Predicted: {predicted_fruit.upper()}\nConfidence: {confidence:.1f}%", 
                     fontsize=18, fontweight='bold', color='darkgreen')
            plt.axis('off')
            
            if save_result:
                result_name = f"result_{predicted_fruit}_{confidence:.0f}percent.png"
                plt.savefig(result_name, bbox_inches='tight', dpi=150)
                print(f"Result saved: {result_name}")
            
            plt.show()
        
        return predicted_fruit, confidence
    
    def batch_predict(self, folder_path):
        """Predict all images in a folder"""
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            return
        
        print(f"BATCH PREDICTION: {folder_path}")
        print("-" * 50)
        
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(folder_path, file)
                self.predict_fruit(img_path, show_image=False)
                print()
    
    def show_dataset_status(self):
        """Show current dataset information"""
        print("DATASET STATUS:")
        print("-" * 30)
        
        total = 0
        for fruit in self.fruits:
            fruit_dir = os.path.join('dataset', fruit)
            if os.path.exists(fruit_dir):
                count = len([f for f in os.listdir(fruit_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                total += count
                status = "OK" if count >= 5 else "LOW" if count > 0 else "EMPTY"
                print(f"{status} {fruit}: {count} images")
        
        print(f"\nTotal: {total} images")
    
    def cleanup_files(self):
        """Delete unused and empty files/folders"""
        print("CLEANING UP FILES...")
        
        # Delete old unused files
        unused_files = [
            'rc.py', 'setup_dataset.py', 'add_images.py', 'create_samples.py',
            'download_real_images.py', 'auto_predict.py', 'simple_predict.py',
            'test_mango.py', 'check_dataset.py', 'fruit_model_auto.h5'
        ]
        
        deleted_count = 0
        for file in unused_files:
            if os.path.exists(file):
                os.remove(file)
                print(f"Deleted: {file}")
                deleted_count += 1
        
        # Remove empty folders
        if os.path.exists('new_images'):
            for fruit in self.fruits:
                fruit_dir = os.path.join('new_images', fruit)
                if os.path.exists(fruit_dir) and len(os.listdir(fruit_dir)) == 0:
                    os.rmdir(fruit_dir)
                    print(f"Removed empty: new_images/{fruit}")
                    deleted_count += 1
            
            if os.path.exists('new_images') and len(os.listdir('new_images')) == 0:
                os.rmdir('new_images')
                print("Removed empty: new_images folder")
                deleted_count += 1
        
        print(f"\nCleanup complete! Deleted {deleted_count} items")

def main():
    """Main menu system"""
    classifier = FruitClassifier()
    
    while True:
        print("\n" + "="*50)
        print("FRUIT CLASSIFICATION SYSTEM")
        print("="*50)
        print("1. Train new model")
        print("2. Load existing model")
        print("3. Predict single image")
        print("4. Batch predict folder")
        print("5. Show dataset status")
        print("6. Quick test (auto)")
        print("7. Cleanup unused files")
        print("0. Exit")
        
        choice = input("\nEnter choice: ").strip()
        
        if choice == "1":
            classifier.train_model()
        
        elif choice == "2":
            if not classifier.load_model():
                print("No saved model found. Train first!")
        
        elif choice == "3":
            if not classifier.model:
                print("Load model first!")
                continue
            img_path = input("Enter image path: ").strip()
            if os.path.exists(img_path):
                save = input("Save result? (y/n): ").lower() == 'y'
                classifier.predict_fruit(img_path, save_result=save)
            else:
                print("Image not found!")
        
        elif choice == "4":
            if not classifier.model:
                print("Load model first!")
                continue
            folder = input("Enter folder path: ").strip()
            classifier.batch_predict(folder)
        
        elif choice == "5":
            classifier.show_dataset_status()
        
        elif choice == "6":
            if not classifier.model:
                classifier.load_model() or classifier.train_model()
            
            # Auto test with available images
            test_images = []
            for fruit in classifier.fruits:
                fruit_dir = f"dataset/{fruit}"
                if os.path.exists(fruit_dir):
                    files = [f for f in os.listdir(fruit_dir) if f.endswith('.jpg')]
                    if files:
                        test_images.append(os.path.join(fruit_dir, files[0]))
            
            if test_images:
                test_img = test_images[0]
                print(f"Auto testing with: {test_img}")
                classifier.predict_fruit(test_img, save_result=True)
            else:
                print("No test images found!")
        
        elif choice == "7":
            classifier.cleanup_files()
        
        elif choice == "0":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()