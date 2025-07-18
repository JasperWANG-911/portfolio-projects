import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf

class SimpleRoofClassifier:
    def __init__(self, img_size=(128, 128)):
        self.img_size = img_size
        self.model = None
        self.class_names = ['A', 'B', 'C']  # A:Very New, B:Normal C: Very Old
    
    def build_model(self):
        """Build CNN model with appropriate regularization to prevent overfitting"""
        model = Sequential([
            # First convolution block
            Conv2D(16, (3, 3), activation='relu', padding='same', 
                   input_shape=(*self.img_size, 3),
                   kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Second convolution block
            Conv2D(32, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.3),
            
            # Third convolution block
            Conv2D(64, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            Flatten(),
            Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(len(self.class_names), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def load_images(self, data_dir, masks_dir=None, require_masks=False):
        """Load images and labels with optional mask application"""
        images = []
        labels = []
        image_paths = []
        mask_applied = []
        
        # Count samples per class to check for imbalance
        class_counts = {class_name: 0 for class_name in self.class_names}
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Directory does not exist: {class_dir}")
                continue
                
            print(f"Loading images from {class_dir}...")
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(class_dir, img_file)
                    mask_found = False
                    
                    # Read the image
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Cannot read image: {img_path}")
                        continue
                        
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.img_size)
                    
                    # Apply mask if provided
                    if masks_dir:
                        base_name = os.path.splitext(img_file)[0]
                        possible_mask_extensions = ['.png', '.jpg', '.jpeg']
                        
                        for ext in possible_mask_extensions:
                            mask_file = base_name + ext
                            mask_path = os.path.join(masks_dir, class_name, mask_file)
                            
                            if os.path.exists(mask_path):
                                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                                if mask is None:
                                    print(f"Cannot read mask: {mask_path}")
                                    continue
                                    
                                mask = cv2.resize(mask, self.img_size)
                                mask = mask > 128
                                
                                mask_3ch = np.stack([mask] * 3, axis=2)
                                img = img * mask_3ch
                                mask_found = True
                                break
                        
                        if require_masks and not mask_found:
                            print(f"Mask not found for {img_path}, skipping this image.")
                            continue
                        elif not mask_found:
                            print(f"Mask not found for {img_path}, using original image.")
                    
                    img = img / 255.0
                    
                    images.append(img)
                    labels.append(class_idx)
                    image_paths.append(img_path)
                    mask_applied.append(mask_found)
                    class_counts[class_name] += 1
        
        print(f"Loaded {len(images)} images, with masks applied to {sum(mask_applied)} images.")
        print(f"Class distribution: {class_counts}")
        
        if len(images) == 0:
            raise ValueError("No images were loaded. Please check your dataset paths and formats.")
            
        return np.array(images), np.array(labels), image_paths
    
    def train(self, images, labels, validation_split=0.2, epochs=50, batch_size=16):
        """Train the model without data augmentation"""
        if self.model is None:
            self.build_model()
        
        labels_onehot = to_categorical(labels, num_classes=len(self.class_names))
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels_onehot, test_size=validation_split, random_state=42, stratify=labels
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        
        # Calculate class weights to handle imbalance
        class_weights = {}
        n_samples = len(labels)
        n_classes = len(self.class_names)
        for i in range(n_classes):
            n_samples_class = np.sum(labels == i)
            class_weights[i] = n_samples / (n_classes * n_samples_class)
        
        print(f"Class weights: {class_weights}")
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train without augmentation
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Evaluate on validation set
        val_loss, val_acc = self.model.evaluate(X_val, y_val, verbose=1)
        print(f"Final validation accuracy: {val_acc:.4f}")
        
        return history
    
    def save_model(self, path):
        """Save the model"""
        if self.model is not None:
            self.model.save(path)
            print(f"Model saved to: {path}")
        else:
            raise ValueError("No model to save. Please train the model first.")
    
    def load_model(self, path):
        """Load the model"""
        if os.path.exists(path):
            try:
                self.model = load_model(path)
                print(f"Model loaded from: {path}")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        else:
            print(f"Cannot find model at: {path}")
            return False
    
    def predict(self, image_path, mask_path=None):
        """Predict the condition of one roof"""
        if self.model is None:
            raise ValueError("Error: No pre-trained model loaded")
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        
        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Cannot read mask: {mask_path}, using original image.")
            else:
                mask = cv2.resize(mask, self.img_size)
                mask = mask > 128
                
                mask_3ch = np.stack([mask] * 3, axis=2)
                img = img * mask_3ch
        
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        pred = self.model.predict(img)[0]
        pred_class = np.argmax(pred)
        pred_prob = pred[pred_class]
        
        class_probs = {self.class_names[i]: float(pred[i]) for i in range(len(self.class_names))}
        
        return self.class_names[pred_class], pred_prob, class_probs
    
    def evaluate_model(self, test_images, test_labels):
        """Evaluate model on test set and return metrics"""
        if self.model is None:
            raise ValueError("No model to evaluate. Please train or load a model first.")
            
        test_labels_onehot = to_categorical(test_labels, num_classes=len(self.class_names))
        loss, accuracy = self.model.evaluate(test_images, test_labels_onehot, verbose=1)
        
        # Calculate accuracy per class
        predictions = self.model.predict(test_images)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = test_labels
        
        class_accuracies = {}
        confusion_matrix = np.zeros((len(self.class_names), len(self.class_names)), dtype=int)
        
        for i in range(len(true_classes)):
            confusion_matrix[true_classes[i]][pred_classes[i]] += 1
        
        for i, class_name in enumerate(self.class_names):
            class_indices = np.where(true_classes == i)[0]
            if len(class_indices) > 0:
                class_acc = np.mean(pred_classes[class_indices] == i)
                class_accuracies[class_name] = class_acc
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'class_accuracies': class_accuracies,
            'confusion_matrix': confusion_matrix
        }
    
    def visualize_results(self, history, save_path='training_history.png', show_plot=True):
        """Visualize training history"""
        plt.figure(figsize=(12, 5))
        
        # Plot training & validation accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        
        # Plot training & validation loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Training history plot saved to {save_path}")
        
        if show_plot:
            try:
                plt.show()
            except Exception as e:
                print(f"Unable to display plot: {e}")
        
        plt.close()


def main():
    """Main function to train and evaluate the roof classifier"""
    try:
        # User-configurable options
        BATCH_SIZE = 16  # Batch size for training
        EPOCHS = 50      # Maximum number of epochs
        VALIDATION_SPLIT = 0.2  # Proportion of data to use for validation
        
        # Print tensorflow version for debugging
        print(f"TensorFlow version: {tf.__version__}")
        
        # Initialize classifier
        classifier = SimpleRoofClassifier()
        
        # Paths to data
        data_dir = "../ROOF_RANKING/training_dataset(grey roof)/Data"
        masks_dir = "../ROOF_RANKING/training_dataset(grey roof)/Masks"
        model_path = "roof_classifier_model.h5"
        
        # Check if data directory exists
        if not os.path.exists(data_dir):
            print(f"Data directory not found: {data_dir}")
            return
        
        # Load images with masks if available
        try:
            if os.path.exists(masks_dir):
                print(f"Using masks from: {masks_dir}")
                images, labels, _ = classifier.load_images(data_dir, masks_dir)
            else:
                print("No masks directory found, using original images.")
                images, labels, _ = classifier.load_images(data_dir)
            
            # Check image data statistics
            print(f"Image data shape: {images.shape}")
            print(f"Image value range: min={np.min(images)}, max={np.max(images)}, mean={np.mean(images)}")
            
        except ValueError as e:
            print(f"Error loading images: {e}")
            return
        
        # Train the model
        history = classifier.train(
            images, 
            labels, 
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT
        )
        
        # Visualize training results
        classifier.visualize_results(history, save_path="training_history.png", show_plot=False)
        
        # Save the model
        try:
            classifier.save_model(model_path)
        except ValueError as e:
            print(f"Error saving model: {e}")
        
        # Evaluate the model on the entire dataset
        print("\nEvaluating model on the entire dataset:")
        eval_results = classifier.evaluate_model(images, labels)
        print(f"Overall accuracy: {eval_results['accuracy']:.4f}")
        print("Class accuracies:")
        for class_name, acc in eval_results['class_accuracies'].items():
            print(f"  {class_name}: {acc:.4f}")
        
        print("\nConfusion Matrix:")
        conf_matrix = eval_results['confusion_matrix']
        for i in range(len(classifier.class_names)):
            row = " ".join([f"{conf_matrix[i][j]:4d}" for j in range(len(classifier.class_names))])
            print(f"{classifier.class_names[i]}: {row}")
        
        # Test on a specific image if available
        test_image = "test_image.jpg"
        test_mask = "test_mask.png"
        
        if os.path.exists(test_image):
            try:
                # Make prediction
                mask_path = test_mask if os.path.exists(test_mask) else None
                pred_class, pred_prob, all_probs = classifier.predict(test_image, mask_path)
                print(f"\nPrediction result: {pred_class} (condition), probability: {pred_prob:.2f}")
                print("Class probabilities:", all_probs)
                print("A: Very New, B: Normal, C: Very Old")
            except Exception as e:
                print(f"Error making prediction: {e}")
    
    except Exception as e:
        import traceback
        print(f"An error occurred during execution: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()