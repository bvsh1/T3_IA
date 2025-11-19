import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os

# ==========================================
# CONFIGURACIÓN E HIPERPARÁMETROS
# Se deben justificar en el informe
# ==========================================
BATCH_SIZE = 32       # Tamaño del lote para actualización de pesos
LEARNING_RATE = 0.001 # Tasa de aprendizaje para el optimizador Adam
EPOCHS = 15           # Número de pasadas completas al dataset
IMG_SIZE = 64         # Tamaño indicado en el PDF 
DROPOUT_RATE = 0.5    # Probabilidad de apagar neuronas para evitar Overfitting

# Configuración de dispositivo (GPU o CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# ==========================================
# 1. PREPARACIÓN DE DATOS
# Conjuntos disjuntos: Train, Val, Test
# ==========================================
def get_data_loaders(data_dir='./data'):
    # Transformaciones: Redimensionar a 64x64, convertir a tensor y normalizar
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Cargar dataset completo
    try:
        full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    except FileNotFoundError:
        print(f"ERROR: No se encontró la carpeta '{data_dir}'. Verifica la ruta.")
        return None, None, None, None

    classes = full_dataset.classes
    print(f"Clases encontradas: {classes}")

    # División: 70% Train, 15% Val, 15% Test
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])

    # DataLoaders
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Datos cargados: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
    return train_loader, val_loader, test_loader, classes

# ==========================================
# 2. ARQUITECTURAS DE CNN
# Capas convolucionales, pooling y FCL
# ==========================================

# Arquitectura Base
class BaseCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(BaseCNN, self).__init__()
        # Bloque 1: 64x64 -> 32x32
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Bloque 2: 32x32 -> 16x16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Bloque 3: 16x16 -> 8x8
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Fully Connected
        self.flatten = nn.Flatten()
        # Entrada a FC: 64 canales * 8 * 8 dimensiones espaciales
        self.fc1 = nn.Linear(64 * 8 * 8, 128) 
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x

# Arquitectura con Dropout 
class DropoutCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(DropoutCNN, self).__init__()
        # Reutilizamos la estructura convolucional
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT_RATE), # Capa de Dropout añadida aquí
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==========================================
# FUNCIONES DE ENTRENAMIENTO Y PLOTEO
# Curvas de entrenamiento y validación
# ==========================================
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    return running_loss / len(loader), 100 * correct / total

def validate_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return running_loss / len(loader), 100 * correct / total

def train_complete_model(model, train_loader, val_loader, name="Modelo"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    print(f"\nIniciando entrenamiento: {name}")
    for epoch in range(EPOCHS):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc = validate_epoch(model, val_loader, criterion)
        
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {t_loss:.4f} Acc: {t_acc:.2f}% | Val Loss: {v_loss:.4f} Acc: {v_acc:.2f}%")
        
    return history

def plot_history(history, title):
    epochs_range = range(1, EPOCHS + 1)
    plt.figure(figsize=(12, 5))
    
    # Gráfico de Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], label='Val Loss', linestyle='--')
    plt.title(f'{title} - Loss')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    
    # Gráfico de Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Train Acc')
    plt.plot(epochs_range, history['val_acc'], label='Val Acc', linestyle='--')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 3. EVALUACIÓN FINAL
# Comparación de desempeño con métricas
# ==========================================
def evaluate_test_set(model, test_loader, classes, name="Modelo"):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    print(f"\n--- Reporte de Clasificación: {name} ---")
    # Métrica 1 y 2: Precision, Recall, F1-score incluidos en el reporte
    print(classification_report(y_true, y_pred, target_names=classes))
    
    print(f"--- Matriz de Confusión: {name} ---")
    print(confusion_matrix(y_true, y_pred))

# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == '__main__':
    # 1. Cargar datos
    train_loader, val_loader, test_loader, classes = get_data_loaders('./data')
    
    if train_loader:
        # 2. Entrenar y evaluar Base CNN
        base_model = BaseCNN(num_classes=len(classes)).to(device)
        hist_base = train_complete_model(base_model, train_loader, val_loader, name="Base CNN")
        plot_history(hist_base, "Base CNN")
        evaluate_test_set(base_model, test_loader, classes, name="Base CNN")
        
        # 3. Entrenar y evaluar Dropout CNN
        dropout_model = DropoutCNN(num_classes=len(classes)).to(device)
        hist_drop = train_complete_model(dropout_model, train_loader, val_loader, name="Dropout CNN")
        plot_history(hist_drop, "Dropout CNN")
        evaluate_test_set(dropout_model, test_loader, classes, name="Dropout CNN")