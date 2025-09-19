import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
iris = load_iris()
X = iris.data[:, [2, 3]] 
y = iris.target
feature_names_2D = [iris.feature_names[2], iris.feature_names[3]]
target_names = iris.target_names
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
print("--- Data Split (Stratified) ---")
print(f"Training set size: {X_train.shape[0]} samples (60%)")
print(f"Test set size: {X_test.shape[0]} samples (40%)")
print("-" * 20)
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)
print("✅ SVM Model trained successfully!")
print("-" * 20)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))
plt.figure(figsize=(10, 7))
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=y_train,
                palette='viridis', s=80, alpha=0.8, edgecolor='k',
                marker='o')
sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=y_test,
                palette='viridis', s=150, alpha=0.9, edgecolor='red',
                marker='X')

plt.xlabel(feature_names_2D[0])
plt.ylabel(feature_names_2D[1])
plt.title('SVM Decision Boundaries (Stratified Split)')
plt.legend(handles=plt.gca().get_legend_handles_labels()[0], labels=list(target_names), title="Species")
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('decision_boundary_final.png')
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names,
            annot=False)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        color = "white" if cm[i, j] > (cm.max() / 2) else "black"
        plt.text(j + 0.5, i + 0.5, str(cm[i, j]),
                 ha='center', va='center',
                 color=color, fontsize=14)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Final Confusion Matrix with All Numbers')
plt.savefig('confusion_matrix_final.png')
plt.show()