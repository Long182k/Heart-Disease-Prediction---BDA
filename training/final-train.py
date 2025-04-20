import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

# Reproducibility
manualSeed = 2019
random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    print("worker_seed", worker_seed)

g = torch.Generator().manual_seed(manualSeed)

# Define transform (no augmentation)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Load dataset
train_dir = "/kaggle/input/combined-isic-2019-2020-annotated-images/Combined_Training_By_Class"
dataset = datasets.ImageFolder(root=train_dir, transform=transform)
print("load dataset done", len(dataset))

class_names = dataset.classes
print("class_names", class_names)
class_to_idx = dataset.class_to_idx
print("Danh sách class và chỉ số tương ứng:", class_to_idx)

# K-fold cross-validation (5 folds)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
skf = StratifiedKFold(n_splits=5)
labels = [label for _, label in dataset]
print("K-fold cross-validation done", len(labels), "labels")

# Build model architecture
class CNN_VGG8(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.conv4 = nn.Conv2d(256, 512, 3)
        self.conv5 = nn.Conv2d(512, 512, 3)
        self.fc1 = nn.Linear(2048, 768)
        self.fc2 = nn.Linear(768, 512)
        self.fc3 = nn.Linear(512, 9)
    
    def forward(self, X):
        out = F.max_pool2d(F.relu(self.conv1(X)), kernel_size=2, stride=2)
        out = F.max_pool2d(F.relu(self.conv2(out)), kernel_size=2, stride=2)
        out = F.max_pool2d(F.relu(self.conv3(out)), kernel_size=2, stride=2)
        out = F.max_pool2d(F.relu(self.conv4(out)), kernel_size=2, stride=2)
        out = F.max_pool2d(F.relu(self.conv5(out)), kernel_size=2, stride=2)
        out = torch.flatten(out, 1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

# Function to plot and save confusion matrix
def plot_confusion_matrix(cm, class_names, title, filename):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(filename)
    plt.close()

# Trainer function with metrics and confusion matrix per epoch
def trainer(model, epochs, train_data, val_data, fold_idx):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005)
    best_accuracy_val = 0
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        correct = 0
        total = 0
        total_loss = 0.0
        train_label_list = []
        train_predict_list = []
        for i, data in enumerate(train_data):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(inputs)
            _, predicted = torch.max(out, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_label_list.extend(labels.cpu().numpy())
            train_predict_list.extend(predicted.cpu().numpy())
        
        accuracy_train = 100 * correct / total
        train_report = classification_report(train_label_list, train_predict_list, output_dict=True, zero_division=0)
        train_precision = train_report['weighted avg']['precision'] * 100
        train_recall = train_report['weighted avg']['recall'] * 100
        train_f1 = train_report['weighted avg']['f1-score'] * 100
        
        # Compute and save training confusion matrix
        train_cm = confusion_matrix(train_label_list, train_predict_list)
        plot_confusion_matrix(
            train_cm, class_names,
            title=f'Confusion Matrix (Train) - Fold {fold_idx + 1}, Epoch {epoch}',
            filename=f'confusion_matrix_fold_{fold_idx + 1}_epoch_{epoch}_train.png'
        )
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        val_label_list = []
        val_predict_list = []
        val_probs_list = []
        with torch.no_grad():
            for i, data in enumerate(val_data):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                out = model(inputs)
                probs = F.softmax(out, dim=1)
                _, predicted = torch.max(out, 1)
                val_label_list.extend(labels.cpu().numpy())
                val_predict_list.extend(predicted.cpu().numpy())
                val_probs_list.extend(probs.cpu().numpy())
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        accuracy_val = 100 * correct / total
        val_report = classification_report(val_label_list, val_predict_list, output_dict=True, zero_division=0)
        val_precision = val_report['weighted avg']['precision'] * 100
        val_recall = val_report['weighted avg']['recall'] * 100
        val_f1 = val_report['weighted avg']['f1-score'] * 100
        
        # Compute and save validation confusion matrix
        val_cm = confusion_matrix(val_label_list, val_predict_list)
        plot_confusion_matrix(
            val_cm, class_names,
            title=f'Confusion Matrix (Validation) - Fold {fold_idx + 1}, Epoch {epoch}',
            filename=f'confusion_matrix_fold_{fold_idx + 1}_epoch_{epoch}_val.png'
        )
        
        # Log metrics for the epoch
        print(f"Epoch {epoch}: "
              f"Loss = {total_loss / len(train_data):.4f}, "
              f"Train Accuracy = {accuracy_train:.2f}%, "
              f"Train Precision = {train_precision:.2f}%, "
              f"Train Recall = {train_recall:.2f}%, "
              f"Train F1 = {train_f1:.2f}%, "
              f"Val Accuracy = {accuracy_val:.2f}%, "
              f"Val Precision = {val_precision:.2f}%, "
              f"Val Recall = {val_recall:.2f}%, "
              f"Val F1 = {val_f1:.2f}%")
        
        if accuracy_val > best_accuracy_val:
            best_accuracy_val = accuracy_val
    
    # Final validation metrics
    final_report = classification_report(val_label_list, val_predict_list, output_dict=True, zero_division=0)
    final_conf_matrix = confusion_matrix(val_label_list, val_predict_list)
    
    # Save final validation confusion matrix
    plot_confusion_matrix(
        final_conf_matrix, class_names,
        title=f'Final Confusion Matrix (Validation) - Fold {fold_idx + 1}',
        filename=f'final_confusion_matrix_fold_{fold_idx + 1}_val.png'
    )
    
    # ROC Curve for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(np.array(val_label_list) == i, np.array(val_probs_list)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    return (accuracy_train, best_accuracy_val, total_loss / len(train_data), final_report,
            final_conf_matrix, fpr, tpr, roc_auc, val_label_list, val_predict_list)

# Main loop
model_list = [CNN_VGG8() for _ in range(5)]
fold_metrics = defaultdict(list)
best_val = 0
best_fold = 0

for i_fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    
    fold_train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, worker_init_fn=seed_worker, generator=g)
    fold_val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, worker_init_fn=seed_worker, generator=g)
    
    print(f"\nFold {i_fold + 1}: Training...")
    
    # Train the model
    (accuracy_train, accuracy_val, loss, report, conf_matrix, fpr, tpr, roc_auc,
     val_label_list, val_predict_list) = trainer(
        model=model_list[i_fold].to(device),
        epochs=30,
        train_data=fold_train_dataloader,
        val_data=fold_val_dataloader,
        fold_idx=i_fold
    )
    
    # Store metrics
    fold_metrics['accuracy_train'].append(accuracy_train)
    fold_metrics['accuracy_val'].append(accuracy_val)
    fold_metrics['precision'].append(report['weighted avg']['precision'] * 100)
    fold_metrics['recall'].append(report['weighted avg']['recall'] * 100)
    fold_metrics['f1_score'].append(report['weighted avg']['f1-score'] * 100)
    fold_metrics['confusion_matrix'].append(conf_matrix)
    fold_metrics['roc_auc'].append(roc_auc)
    
    # Print fold metrics
    print(f"\nFold {i_fold + 1} Metrics:")
    print(f"Train Accuracy: {accuracy_train:.2f}%")
    print(f"Val Accuracy: {accuracy_val:.2f}%")
    print(f"Precision: {report['weighted avg']['precision'] * 100:.2f}%")
    print(f"Recall: {report['weighted avg']['recall'] * 100:.2f}%")
    print(f"F1 Score: {report['weighted avg']['f1-score'] * 100:.2f}%")
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", classification_report(val_label_list, val_predict_list, target_names=class_names, zero_division=0))
    
    # Plot ROC Curve for each class
    plt.figure(figsize=(10, 8))
    for i in range(len(class_names)):
        plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Fold {i_fold + 1}')
    plt.legend(loc="lower right")
    plt.savefig(f'roc_curve_fold_{i_fold + 1}.png')
    plt.close()
    
    # Save the best model
    if accuracy_val > best_val:
        best_val = accuracy_val
        best_fold = i_fold
        torch.save(model_list[i_fold].state_dict(), '/kaggle/working/best_cancer_prediction.pth')

# Final summary of folds
print("\nFinal Summary of All Folds:")
for i in range(5):
    print(f"\nFold {i + 1}:")
    print(f"Train Accuracy: {fold_metrics['accuracy_train'][i]:.2f}%")
    print(f"Val Accuracy: {fold_metrics['accuracy_val'][i]:.2f}%")
    print(f"Precision: {fold_metrics['precision'][i]:.2f}%")
    print(f"Recall: {fold_metrics['recall'][i]:.2f}%")
    print(f"F1 Score: {fold_metrics['f1_score'][i]:.2f}%")

print(f"\nThe best model is from fold {best_fold + 1} with validation accuracy {best_val:.2f}%")

# Evaluate on new test set
print("\nEvaluating on New Test Set...")
test_dir = "/kaggle/input/mapping-annotated-test-isic-2019-by-classes/ISIC_2019_Test_By_Class"
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, worker_init_fn=seed_worker, generator=g)

best_model = CNN_VGG8().to(device)
best_model.load_state_dict(torch.load("/kaggle/working/best_cancer_prediction.pth", weights_only=True))
best_model.eval()

correct = 0
total = 0
test_label_list = []
test_predict_list = []
test_probs_list = []

with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        out = best_model(inputs)
        probs = F.softmax(out, dim=1)
        _, predicted = torch.max(out, 1)
        test_label_list.extend(labels.cpu().numpy())
        test_predict_list.extend(predicted.cpu().numpy())
        test_probs_list.extend(probs.cpu().numpy())
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

# Compute test set metrics
test_accuracy = 100 * correct / total
test_report = classification_report(test_label_list, test_predict_list, target_names=class_names, output_dict=True, zero_division=0)
test_conf_matrix = confusion_matrix(test_label_list, test_predict_list)

# Save final test confusion matrix
plot_confusion_matrix(
    test_conf_matrix, class_names,
    title='Final Confusion Matrix (Test Set)',
    filename='final_confusion_matrix_test.png'
)

# ROC Curve for test set
test_fpr = {}
test_tpr = {}
test_roc_auc = {}
for i in range(len(class_names)):
    test_fpr[i], test_tpr[i], _ = roc_curve(np.array(test_label_list) == i, np.array(test_probs_list)[:, i])
    test_roc_auc[i] = auc(test_fpr[i], test_tpr[i])

# Print test set metrics
print(f"\nTest Set Metrics:")
print(f"Accuracy: {test_accuracy:.2f}%")
print(f"Precision: {test_report['weighted avg']['precision'] * 100:.2f}%")
print(f"Recall: {test_report['weighted avg']['recall'] * 100:.2f}%")
print(f"F1 Score: {test_report['weighted avg']['f1-score'] * 100:.2f}%")
print("Confusion Matrix:\n", test_conf_matrix)
print("Classification Report:\n", classification_report(test_label_list, test_predict_list, target_names=class_names, zero_division=0))

# Plot ROC Curve for test set
plt.figure(figsize=(10, 8))
for i in range(len(class_names)):
    plt.plot(test_fpr[i], test_tpr[i], label=f'{class_names[i]} (AUC = {test_roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Test Set')
plt.legend(loc="lower right")
plt.savefig('roc_curve_test_set.png')
plt.close()