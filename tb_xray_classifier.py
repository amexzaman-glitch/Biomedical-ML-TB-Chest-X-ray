# ============================================
# COMPREHENSIVE TB CHEST X-RAY CLASSIFICATION
# With Biomedical Interpretation
# ============================================

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, \
    precision_recall_curve
import joblib

print("🩺 TUBERCULOSIS DETECTION SYSTEM")
print("=" * 50)

# ==================== 1. LOAD DATASET ====================
print("\n📁 STEP 1: Loading Dataset...")


def load_medical_dataset(dataset_path):
    """Load and organize medical image dataset"""
    print(f"📂 Loading from: {dataset_path}")

    if not os.path.exists(dataset_path):
        print("❌ Dataset not found!")
        return None, None, None, None

    def load_images_from_folder(folder_path, label):
        images, labels = [], []
        if os.path.exists(folder_path):
            for img_file in os.listdir(folder_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(folder_path, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        images.append(img)
                        labels.append(label)
        return images, labels

    # Load both classes
    normal_images, normal_labels = load_images_from_folder(
        os.path.join(dataset_path, "Normal"), 0)
    tb_images, tb_labels = load_images_from_folder(
        os.path.join(dataset_path, "Tuberculosis"), 1)

    print(f"✅ Dataset loaded:")
    print(f"   🏥 Normal cases: {len(normal_images)}")
    print(f"   🦠 TB cases: {len(tb_images)}")
    print(f"   📊 Total: {len(normal_images) + len(tb_images)} images")

    # Return both combined and separate datasets
    all_images = normal_images + tb_images
    all_labels = normal_labels + tb_labels
    return all_images, all_labels, normal_images, tb_images


# Load dataset
dataset_path = r"C:\Users\AMAN12\Desktop\archive (4)\TB_Chest_Radiography_Database"
images, labels, normal_images, tb_images = load_medical_dataset(dataset_path)

if images is None:
    exit()

# ==================== 2. PREPROCESSING ====================
print("\n🔧 STEP 2: Preprocessing Images...")


def preprocess_images(images, target_size=(64, 64)):
    """Normalize and prepare images for ML"""
    processed = []
    print("🔄 Preprocessing images...")

    for img in images:
        # Resize to consistent dimensions
        img_resized = cv2.resize(img, target_size)
        # Normalize pixel values to 0-1
        img_normalized = img_resized / 255.0
        # Flatten for traditional ML
        img_flattened = img_normalized.flatten()
        processed.append(img_flattened)

    print(f"✅ Preprocessing complete:")
    print(f"   📐 Image size: {target_size}")
    print(f"   🎯 Features per image: {len(processed[0])}")

    return np.array(processed)


X = preprocess_images(images)
y = np.array(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 Data split:")
print(f"   🎓 Training: {X_train.shape[0]} images")
print(f"   🧪 Testing: {X_test.shape[0]} images")
print(f"   ⚖️ Class balance: {y_train.mean():.3f} TB ratio")

# ==================== 3. TRAIN MODEL ====================
print("\n🤖 STEP 3: Training Model...")

# Initialize Random Forest classifier
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

print("🚀 Training Random Forest classifier...")
model.fit(X_train, y_train)
print("✅ Model training completed!")

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("📝 Predictions generated on test set")

# ==================== 4. EVALUATE METRICS ====================
print("\n📈 STEP 4: Comprehensive Evaluation...")

# Calculate all metrics
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Clinical metrics
sensitivity = tp / (tp + fn)  # Recall
specificity = tn / (tn + fp)
precision = tp / (tp + fp)
f1 = 2 * (precision * sensitivity) / (precision + sensitivity)

# ROC analysis
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Precision-Recall analysis
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall_curve, precision_curve)

print("\n" + "=" * 60)
print("🎯 TECHNICAL PERFORMANCE METRICS")
print("=" * 60)
print(f"📊 Accuracy:    {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"❤️  Sensitivity: {sensitivity:.4f} ({sensitivity * 100:.2f}%)")
print(f"💙 Specificity: {specificity:.4f} ({specificity * 100:.2f}%)")
print(f"🎯 Precision:   {precision:.4f} ({precision * 100:.2f}%)")
print(f"⚖️  F1-Score:    {f1:.4f} ({f1 * 100:.2f}%)")
print(f"📈 ROC AUC:     {roc_auc:.4f}")
print(f"📊 PR AUC:      {pr_auc:.4f}")

print(f"\n🔢 Confusion Matrix:")
print(f"                Predicted")
print(f"              Normal    TB")
print(f"Actual Normal:  {tn:4d}    {fp:4d}")
print(f"Actual TB:      {fn:4d}    {tp:4d}")

# ==================== 5. BIOMEDICAL INTERPRETATION ====================
print("\n" + "=" * 60)
print("🩺 BIOMEDICAL INTERPRETATION")
print("=" * 60)

print(f"\n🏥 CLINICAL PERFORMANCE ASSESSMENT:")

# Sensitivity interpretation
if sensitivity >= 0.95:
    print("✅ SENSITIVITY: EXCELLENT - Very few TB cases missed")
    print("   → Suitable for high-sensitivity screening")
elif sensitivity >= 0.90:
    print("✅ SENSITIVITY: GOOD - Acceptable missed case rate")
    print("   → Can be used with clinical correlation")
else:
    print("⚠️  SENSITIVITY: NEEDS IMPROVEMENT")
    print("   → Too many TB cases would be missed")

# Specificity interpretation
if specificity >= 0.90:
    print("✅ SPECIFICITY: EXCELLENT - Minimal false alarms")
    print("   → Reduces unnecessary patient anxiety")
elif specificity >= 0.85:
    print("✅ SPECIFICITY: GOOD - Manageable false positive rate")
    print("   → Acceptable for clinical workflow")
else:
    print("⚠️  SPECIFICITY: NEEDS IMPROVEMENT")
    print("   → Too many false alarms would occur")

print(f"\n🩺 CLINICAL ERROR ANALYSIS:")
print(f"❌ False Negatives: {fn} patients")
print(f"   → TB cases that would be missed")
print(f"   → RISK: Delayed treatment, disease progression")
print(f"❌ False Positives: {fp} patients")
print(f"   → Healthy people incorrectly flagged")
print(f"   → IMPACT: Unnecessary anxiety, additional testing")

print(f"\n📊 POPULATION IMPACT ANALYSIS:")
total_tests = len(y_test)
print(f"• Tested on {total_tests} cases")
print(f"• Would correctly identify {tp} TB patients for treatment")
print(f"• Would miss {fn} TB cases requiring follow-up")
print(f"• Would cause {fp} unnecessary follow-up tests")

# ==================== 6. VISUALIZE RESULTS ====================
print("\n📊 STEP 5: Creating Medical Visualizations...")

plt.figure(figsize=(20, 15))

# Plot 1: Confusion Matrix
plt.subplot(2, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Normal', 'Predicted TB'],
            yticklabels=['Actual Normal', 'Actual TB'],
            annot_kws={'size': 14, 'weight': 'bold'})
plt.title('Confusion Matrix\nClinical Performance Overview', fontsize=12, fontweight='bold')
plt.ylabel('True Diagnosis', fontweight='bold')
plt.xlabel('AI Prediction', fontweight='bold')

# Plot 2: ROC Curve
plt.subplot(2, 3, 2)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier', alpha=0.5)
plt.fill_between(fpr, tpr, alpha=0.2, color='darkorange')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
plt.ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
plt.title('ROC Curve\nDiagnostic Discrimination Ability', fontsize=12, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

# Plot 3: Precision-Recall Curve
plt.subplot(2, 3, 3)
plt.plot(recall_curve, precision_curve, color='purple', lw=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
plt.fill_between(recall_curve, precision_curve, alpha=0.2, color='purple')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall (Sensitivity)', fontweight='bold')
plt.ylabel('Precision', fontweight='bold')
plt.title('Precision-Recall Curve\nClinical Trade-off Analysis', fontsize=12, fontweight='bold')
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)

# Plot 4: Performance Metrics
plt.subplot(2, 3, 4)
metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1-Score']
values = [accuracy, sensitivity, specificity, precision, f1]
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#4CAF50']

bars = plt.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')
plt.title('Clinical Performance Metrics', fontsize=12, fontweight='bold')
plt.ylabel('Score', fontweight='bold')
plt.ylim(0, 1.1)
plt.xticks(rotation=45, ha='right')

# Add clinical threshold lines
plt.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Excellent')
plt.axhline(y=0.90, color='orange', linestyle='--', alpha=0.7, label='Good')

for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.legend()

# Plot 5: Feature Importance
plt.subplot(2, 3, 5)
feature_importance = model.feature_importances_
top_n = 20
top_indices = np.argsort(feature_importance)[-top_n:]
top_importance = feature_importance[top_indices]

plt.barh(range(len(top_importance)), top_importance, color='lightseagreen', alpha=0.7)
plt.title(f'Top {top_n} Most Important\nImage Regions', fontsize=12, fontweight='bold')
plt.xlabel('Importance Score', fontweight='bold')
plt.yticks(range(len(top_importance)), [f'Region {i + 1}' for i in range(len(top_importance))])

# Plot 6: Sample Images (FIXED - using the separate normal_images and tb_images)
plt.subplot(2, 3, 6)
if normal_images and tb_images:  # Check if we have both types of images
    # Display a sample normal image
    normal_sample = cv2.resize(normal_images[0], (64, 64)) if len(normal_images) > 0 else None
    if normal_sample is not None:
        plt.imshow(normal_sample, cmap='gray')
        plt.title('Sample: Normal Chest X-ray', fontsize=11, fontweight='bold')
        plt.axis('off')
        plt.text(0.5, -0.1, 'Clear lung fields\nNormal anatomy',
                 transform=plt.gca().transAxes, ha='center', fontsize=9, style='italic')
else:
    # Fallback: show a message if no sample images available
    plt.text(0.5, 0.5, 'Sample images\nnot available',
             ha='center', va='center', transform=plt.gca().transAxes,
             fontsize=12, fontweight='bold')
    plt.axis('off')

plt.tight_layout()
plt.savefig('medical_tb_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Medical visualizations saved as 'medical_tb_analysis_comprehensive.png'")

# ==================== 7. MODEL PERFORMANCE EVALUATION ====================
print("\n" + "=" * 60)
print("🔍 MODEL PERFORMANCE EVALUATION")
print("=" * 60)

print(f"\n📋 Classification Report:")
print("=" * 40)
print(classification_report(y_test, y_pred, target_names=['Normal', 'Tuberculosis']))

# Clinical deployment recommendation
print(f"\n🎯 CLINICAL DEPLOYMENT ASSESSMENT:")
if sensitivity > 0.95 and specificity > 0.90:
    print("🚀 RECOMMENDATION: SUITABLE FOR CLINICAL USE")
    print("   • High sensitivity for TB detection")
    print("   • Low false positive rate")
    print("   • Can assist radiologists in screening")
elif sensitivity > 0.90 and specificity > 0.85:
    print("✅ RECOMMENDATION: SUITABLE FOR TRIAGE")
    print("   • Good detection capability")
    print("   • Use as second reader")
    print("   • Maintain radiologist oversight")
else:
    print("⚠️  RECOMMENDATION: NEEDS IMPROVEMENT")
    print("   • Consider more training data")
    print("   • Explore different algorithms")
    print("   • Not ready for clinical deployment")

# Save model and results
print(f"\n💾 Saving model and results...")
joblib.dump(model, 'tb_chest_xray_model.pkl')

results = {
    'performance_metrics': {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    },
    'clinical_analysis': {
        'false_negatives': fn,
        'false_positives': fp,
        'deployment_recommendation': 'SUITABLE_FOR_CLINICAL_USE' if sensitivity > 0.95 and specificity > 0.90 else 'SUITABLE_FOR_TRIAGE' if sensitivity > 0.90 and specificity > 0.85 else 'NEEDS_IMPROVEMENT'
    }
}

joblib.dump(results, 'clinical_performance_results.pkl')

print("✅ Model saved: 'tb_chest_xray_model.pkl'")
print("✅ Results saved: 'clinical_performance_results.pkl'")

# ==================== 8. USER INTERACTION - TEST NEW IMAGES ====================
print("\n" + "=" * 60)
print("🔍 USER INTERACTION - TEST YOUR OWN X-RAY IMAGES")
print("=" * 60)


def predict_single_image(image_path):
    """Predict whether a single X-ray image shows TB or Normal"""
    try:
        # Load and preprocess the image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"❌ Could not load image: {image_path}")
            return None

        # Apply same preprocessing as training data
        img_processed = cv2.resize(img, (64, 64)) / 255.0
        img_flat = img_processed.flatten().reshape(1, -1)

        # Make prediction
        prediction = model.predict(img_flat)[0]
        probability = model.predict_proba(img_flat)[0]

        return prediction, probability, img

    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return None


def interactive_prediction():
    """Interactive interface for users to test their images"""
    print("\n🎯 HOW TO USE:")
    print("1. Place your X-ray image in the same folder as this script")
    print("2. Enter the filename when prompted")
    print("3. Get instant TB detection results with medical interpretation")
    print("4. Type 'quit' to exit\n")

    while True:
        print("-" * 50)
        image_path = input("📤 Enter X-ray image filename (or 'quit' to exit): ").strip()

        if image_path.lower() == 'quit':
            print("👋 Thank you for using the TB Detection System!")
            break

        # Check if file exists
        if not os.path.exists(image_path):
            print(f"❌ File not found: {image_path}")
            print("💡 Make sure the file is in the same folder as this script")
            continue

        # Make prediction
        print(f"\n🔄 Analyzing {image_path}...")
        result = predict_single_image(image_path)

        if result is not None:
            prediction, probability, img = result

            # Display results
            print("\n" + "=" * 50)
            print("🎯 DIAGNOSTIC RESULTS")
            print("=" * 50)

            if prediction == 1:
                print("🔴 DIAGNOSIS: TUBERCULOSIS DETECTED")
                print(f"🟢 Confidence: {probability[1] * 100:.2f}%")
                print(f"🟡 Normal Probability: {probability[0] * 100:.2f}%")
                print("\n⚠️  CLINICAL RECOMMENDATION:")
                print("• Consult a healthcare professional immediately")
                print("• Confirm with additional tests (sputum test, culture)")
                print("• Begin appropriate anti-TB treatment if confirmed")
                print("• Isolate to prevent transmission until evaluated")
            else:
                print("🟢 DIAGNOSIS: NORMAL CHEST X-RAY")
                print(f"🟢 Confidence: {probability[0] * 100:.2f}%")
                print(f"🔴 TB Probability: {probability[1] * 100:.2f}%")
                print("\n✅ CLINICAL NOTE:")
                print("• No signs of tuberculosis detected")
                print("• Continue regular health monitoring")
                print("• Consult doctor if respiratory symptoms develop")

            # Show the analyzed image with results
            plt.figure(figsize=(12, 5))

            # Plot 1: The X-ray image
            plt.subplot(1, 2, 1)
            plt.imshow(img, cmap='gray')
            plt.title(f'Analyzed X-ray: {os.path.basename(image_path)}', fontweight='bold')
            plt.axis('off')

            # Plot 2: Prediction probabilities
            plt.subplot(1, 2, 2)
            labels = ['Normal', 'Tuberculosis']
            colors = ['lightgreen', 'lightcoral']
            plt.bar(labels, probability, color=colors, edgecolor='black', alpha=0.8)
            plt.title('AI Prediction Confidence', fontweight='bold')
            plt.ylabel('Probability')
            plt.ylim(0, 1)

            # Add probability labels on bars
            for i, prob in enumerate(probability):
                plt.text(i, prob + 0.02, f'{prob:.3f}',
                         ha='center', va='bottom', fontweight='bold', fontsize=11)

            plt.tight_layout()
            plt.savefig('user_prediction_result.png', dpi=150, bbox_inches='tight')
            plt.show()

            print(f"\n💾 Results saved as 'user_prediction_result.png'")
            print(
                "💡 Remember: This is an AI-assisted tool. Always consult with healthcare professionals for final diagnosis.")

        else:
            print("❌ Could not process the image. Please try another file.")


# Start interactive prediction
interactive_prediction()
# ==================== FINAL SUMMARY ====================
print("\n" + "=" * 60)
print("🏁 COMPREHENSIVE TB DETECTION ANALYSIS COMPLETE")
print("=" * 60)

print(f"\n📊 FINAL PERFORMANCE SUMMARY:")
print(f"🎯 Overall Accuracy:    {accuracy * 100:.1f}%")
print(f"❤️  TB Detection Rate:   {sensitivity * 100:.1f}%")
print(f"💙 Healthy ID Rate:     {specificity * 100:.1f}%")
print(f"📈 Discrimination:      {roc_auc:.3f} AUC")

print(f"\n🩺 CLINICAL IMPACT:")
print(f"✅ Would correctly identify {tp} TB patients for treatment")
print(f"⚠️  Would miss {fn} TB cases (requires safety protocols)")
print(f"📝 Would generate {fp} false alarms (requires counseling)")

print(f"\n💡 NEXT STEPS:")
print("1. Clinical validation with real patient data")
print("2. Integration with hospital PACS systems")
print("3. Radiologist training and workflow integration")
print("4. Continuous monitoring and model updates")

print("\n" + "=" * 60)
print("Medical AI TB Detection System Ready! 🎉")
print("=" * 60)