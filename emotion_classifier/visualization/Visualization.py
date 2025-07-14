import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

#Loss curve , still don't know list names 
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss per Epoch')
plt.legend()
plt.grid(True)
plt.show()

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Class names
class_names = ['Happy', 'Sad', 'Angry', 'Neutral', 'Surprised'] 

# Plot with seaborn
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()