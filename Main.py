# 18.07.2025 - RR (TensorFlow Version)
# Dieses Skript liest alle TransKI Daten aus einem Ordner, erstellt Zeitfenster für eine beliebige Anzahl von Features und normalisiert diese.
# Anschließend werden verschiedene ML Modelle trainiert (CNN - AE, FFNN - AE, XGBoost, Random Forest) und die Ergebnisse in einer CSV-Datei gespeichert.

import pandas as pd
import numpy as np
from helper_functions import ifw_data
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers, callbacks
import re
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

# Import matplotlib with error handling
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI issues
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
    print("Matplotlib erfolgreich geladen")
except ImportError as e:
    print(f"Matplotlib Import Fehler: {e}")
    print("Plots werden übersprungen")
    PLOTTING_AVAILABLE = False

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def create_windows_and_normalize_improved(data_objects, feature_names, window_size, step_size):
    """
    Verbesserte Normalisierung: Z-Score statt Min-Max für bessere Performance
    Erzeugt normalisierte Zeitfenster für die angegebenen Features aus allen Datenobjekten.
    Normalisierung erfolgt pro Datei (nicht pro Fenster).
    Gibt zusätzlich eine Liste zurück, die für jedes Fenster den Index der Ursprungsdatei enthält.
    """
    windows = []
    file_indices = []
    normalization_params = []  # Speichere Normalisierungsparameter für später
    
    for file_idx, obj in enumerate(data_objects):
        df = obj.data[feature_names].copy()
        arr = df.values
        n = arr.shape[0]
        
        # Z-Score Normalisierung (robuster als Min-Max)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        std[std == 0] = 1  # Division durch 0 vermeiden
        arr_norm = (arr - mean) / std
        
        # Speichere Normalisierungsparameter
        normalization_params.append({'mean': mean, 'std': std})
        
        # Sliding window
        for start in range(0, n - window_size + 1, step_size):
            window = arr_norm[start:start+window_size]
            windows.append(window)
            file_indices.append(file_idx)
    
    return np.array(windows), np.array(file_indices), normalization_params

class ImprovedFeedForwardAutoencoder(Model):
    def __init__(self, input_dim):
        super(ImprovedFeedForwardAutoencoder, self).__init__()
        
        # Berechne sinnvolle Zwischendimensionen
        dim1 = max(input_dim // 2, 64)
        dim2 = max(input_dim // 4, 32)
        dim3 = max(input_dim // 8, 16)
        bottleneck = max(input_dim // 16, 8)
        
        print(f"Autoencoder Architektur: {input_dim} -> {dim1} -> {dim2} -> {dim3} -> {bottleneck}")
        
        # Encoder
        self.encoder_layers = [
            layers.Dense(dim1, activation=None),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.2),
            
            layers.Dense(dim2, activation=None),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.2),
            
            layers.Dense(dim3, activation=None),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            
            layers.Dense(bottleneck, activation=None)
        ]
        
        # Decoder
        self.decoder_layers = [
            layers.Dense(dim3, activation=None),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            
            layers.Dense(dim2, activation=None),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.2),
            
            layers.Dense(dim1, activation=None),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.2),
            
            layers.Dense(input_dim, activation=None)
        ]
        
        # Store architecture info
        self.architecture_info = {
            'input_dim': input_dim,
            'dim1': dim1,
            'dim2': dim2,
            'dim3': dim3,
            'bottleneck': bottleneck
        }

    def call(self, inputs, training=None):
        # Encoder
        x = inputs
        for layer in self.encoder_layers:
            x = layer(x, training=training)
        
        # Store encoded representation
        encoded = x
        
        # Decoder
        for layer in self.decoder_layers:
            x = layer(x, training=training)
        
        return x
    
    def encode(self, inputs, training=None):
        """Return encoded representation"""
        x = inputs
        for layer in self.encoder_layers:
            x = layer(x, training=training)
        return x

def improved_train_autoencoder(X, n_epochs=50, batch_size=256, lr=1e-3):
    """
    Verbessertes Training mit Learning Rate Scheduling, Early Stopping, etc.
    """
    print(f"Starte Training mit {n_epochs} Epochen")
    
    # Check for GPU availability
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"GPU verfügbar: {len(gpus)} GPU(s)")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU Konfiguration fehlgeschlagen: {e}")
    else:
        print("Training auf CPU")
    
    # Prüfe Datenqualität
    print(f"Daten-Shape: {X.shape}")
    print(f"NaN values: {np.isnan(X).sum()}")
    print(f"Inf values: {np.isinf(X).sum()}")
    print(f"Data range: {X.min():.4f} to {X.max():.4f}")
    
    # Train/Validation Split für Early Stopping
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
    
    print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")
    
    # Model erstellen
    model = ImprovedFeedForwardAutoencoder(X.shape[1])
    
    # Build model by calling it once
    dummy_input = tf.random.normal((1, X.shape[1]))
    _ = model(dummy_input)
    
    # Optimizer 
    optimizer = optimizers.AdamW(learning_rate=lr, weight_decay=1e-4)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    # Callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    # Custom callback für Fortschritt
    class ProgressCallback(callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % 5 == 0 or epoch == self.params['epochs'] - 1:
                try:
                    current_lr = float(self.model.optimizer.learning_rate.numpy())
                except:
                    current_lr = lr
                print(f"Epoch {epoch+1}/{self.params['epochs']} - "
                      f"Train Loss: {logs['loss']:.6f}, "
                      f"Val Loss: {logs['val_loss']:.6f}, "
                      f"LR: {current_lr:.2e}")
    
    progress_callback = ProgressCallback()
    
    # Training
    history = model.fit(
        X_train, X_train,  # Autoencoder: input = target
        validation_data=(X_val, X_val),
        epochs=n_epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr, progress_callback],
        verbose=0  # We handle progress with custom callback
    )
    
    return model, {
        'train_losses': history.history['loss'],
        'val_losses': history.history['val_loss'],
        'train_mae': history.history['mae'],
        'val_mae': history.history['val_mae']
    }

def reconstruct_timeseries_from_windows(windows, file_indices, data_objects, window_size, step_size, n_features):
    """
    Rekonstruiert die Zeitreihen pro Datei und Feature aus überlappenden Fenstern (mittelt Überlappungen).
    Verwendet die ursprünglichen Datenlängen aus data_objects.
    """
    n_files = len(data_objects)
    
    # Verwende die tatsächlichen Längen der ursprünglichen Daten
    file_lengths = {i: len(data_objects[i].data) for i in range(n_files)}
    
    # Initialisiere Arrays für Rekonstruktion und Zähler für Mittelung
    timeseries = {i: np.zeros((file_lengths[i], n_features)) for i in range(n_files)}
    counts = {i: np.zeros((file_lengths[i], n_features)) for i in range(n_files)}
    
    # Zähle Fenster pro Datei und erstelle Mapping
    file_window_count = {i: 0 for i in range(n_files)}
    
    for global_win_idx, file_idx in enumerate(file_indices):
        local_win_idx = file_window_count[file_idx]
        start = local_win_idx * step_size
        end = min(start + window_size, file_lengths[file_idx])
        
        if start < file_lengths[file_idx]:
            # Schneide Fenster ab, falls es über die Dateilänge hinausgeht
            window_data = windows[global_win_idx]
            actual_window_size = end - start
            
            if actual_window_size < window_size:
                window_data = window_data[:actual_window_size]
            
            timeseries[file_idx][start:end] += window_data
            counts[file_idx][start:end] += 1
        
        file_window_count[file_idx] += 1
    
    # Mittelung der Überlappungen
    for i in range(n_files):
        mask = counts[i] > 0
        timeseries[i][mask] /= counts[i][mask]
    
    return timeseries

def calculate_reconstruction_error_per_timestep(orig_series, recon_series, feature_names):
    """
    Berechnet den Rekonstruktionsfehler pro Zeitschritt für alle Dateien und Features.
    Gibt DataFrame mit Spalten: file_idx, timestep, feature, error zurück.
    """
    errors = []
    
    for file_idx in orig_series:
        orig = orig_series[file_idx]
        recon = recon_series[file_idx]
        
        for t in range(orig.shape[0]):
            for f_idx, feature in enumerate(feature_names):
                error = abs(orig[t, f_idx] - recon[t, f_idx])
                errors.append({
                    'file_idx': file_idx,
                    'timestep': t,
                    'feature': feature,
                    'error': error
                })
    
    return pd.DataFrame(errors)

def create_plots(training_history, orig_test_series, recon_test_series, feature_names, rms_df, n_train_files, output_dir):
    """
    Erstellt alle Plots und speichert sie als Dateien
    """
    if not PLOTTING_AVAILABLE:
        print("Plots werden übersprungen (matplotlib nicht verfügbar)")
        return
    
    try:
        # 1. Training Loss Plot
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(training_history['train_losses'], label='Training Loss', marker='o', markersize=2)
        plt.plot(training_history['val_losses'], label='Validation Loss', marker='o', markersize=2)
        plt.xlabel('Epoche')
        plt.ylabel('Loss')
        plt.title('Training und Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(training_history['train_losses'], label='Training Loss')
        plt.plot(training_history['val_losses'], label='Validation Loss')
        plt.xlabel('Epoche')
        plt.ylabel('Loss (Log Scale)')
        plt.title('Training und Validation Loss (Log)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Reconstruction Example Plot
        test_file_idx = 0
        feature_idx = 0
        
        if test_file_idx in orig_test_series and feature_idx < len(feature_names):
            orig = orig_test_series[test_file_idx]
            recon = recon_test_series[test_file_idx]
            
            plt.figure(figsize=(15, 8))
            
            # Subplot 1: Vollständige Zeitreihe
            plt.subplot(2, 1, 1)
            plt.plot(orig[:, feature_idx], label='Original', alpha=0.8, linewidth=1)
            plt.plot(recon[:, feature_idx], label='Rekonstruktion', alpha=0.8, linewidth=1)
            plt.title(f'Original vs. Rekonstruktion (Testdatei {test_file_idx+1}, Feature: {feature_names[feature_idx]})')
            plt.xlabel('Zeitindex')
            plt.ylabel('Normierter Wert')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Subplot 2: Zoom auf ersten 1000 Punkte
            plt.subplot(2, 1, 2)
            zoom_end = min(1000, len(orig))
            plt.plot(orig[:zoom_end, feature_idx], label='Original', alpha=0.8, linewidth=1)
            plt.plot(recon[:zoom_end, feature_idx], label='Rekonstruktion', alpha=0.8, linewidth=1)
            plt.title(f'Detail-Ansicht (erste {zoom_end} Punkte)')
            plt.xlabel('Zeitindex')
            plt.ylabel('Normierter Wert')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'reconstruction_example.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. RMS Error Plot
        plt.figure(figsize=(15, 8))
        
        for feat in feature_names:
            subset = rms_df[rms_df['feature'] == feat].sort_values('file_idx')
            plt.plot(subset['file_idx'], subset['rms'], marker='o', linestyle='-', label=feat, markersize=4)
        
        # Linie für Train/Test-Split
        plt.axvline(n_train_files - 0.5, color='red', linestyle='--', linewidth=2, label='Train/Test-Grenze')
        
        plt.xlabel('Datei-Index (Train + Test)')
        plt.ylabel('RMS Fehler')
        plt.title('RMS Rekonstruktionsfehler pro Datei (pro Feature)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rms_errors.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Plots erfolgreich erstellt und gespeichert")
        
    except Exception as e:
        print(f"Fehler beim Erstellen der Plots: {e}")

######################################################
#----------------------------------------------------#
######################## MAIN ########################
#----------------------------------------------------#
######################################################

def main():
    # Verzeichnis mit TransKI-Daten
    print("TensorFlow Version")
    #data_folder = r'D:\\01_Diss\\01_Versuchsdaten\\01_TransKI\\01_Fraesen_Stand\\IfW\\Grob\\Toolox33\\Toolox33_Standzeit_IfW_eID1156'  # ggf. anpassen
    data_folder = r'E:\\01_TransKI_Versuche\\02_STAND_TOOL_FRS\\empolis_onedrive\\Toolox33_Standzeit_IfW_eID1604'
    
    # Alle Dateien im Verzeichnis auflisten
    files = [f for f in os.listdir(data_folder) if f.endswith('.txt') and "Maschinendaten" in f]

    # Sortiere die Dateien nach der Endnummer (z.B. "0001", "0002", ...)
    def extract_file_number(filename):
        # Extrahiere die letzte zusammenhängende Ziffernfolge vor der Dateiendung
        matches = re.findall(r'(\d+)(?=\D*$)', filename)
        return int(matches[-1]) if matches else -1

    files_sorted = sorted(files, key=extract_file_number)

    # Train-Test-Split: 66% Training, Rest Test
    n_total = len(files_sorted)
    n_train = int(n_total * 0.66)
    train_files = files_sorted[:n_train]
    test_files = files_sorted[n_train:]

    print(f"Gefundene Dateien: {n_total}")
    print(f"Training: {len(train_files)}, Test: {len(test_files)}")

    # Liste für ifw_data-Objekte
    train_objects = []
    test_objects = []

    print("Lade Trainingsdateien...")
    for file in train_files:
        try:
            obj = ifw_data(data_folder, file, 'TransKI')
            if obj.header['error'] == "false":
                train_objects.append(obj)
        except Exception as e:
            print(f"Fehler beim Laden von {file}: {e}")

    print("Lade Testdateien...")
    for file in test_files:
        try:
            obj = ifw_data(data_folder, file, 'TransKI')
            if obj.header['error'] == "false":
                test_objects.append(obj)
        except Exception as e:
            print(f"Fehler beim Laden von {file}: {e}")

    print(f"{len(train_objects)} Trainingsdateien, {len(test_objects)} Testdateien erfolgreich geladen.")

    # --- Feature-Abfrage für eine Beispiel-Datei ---
    if len(train_objects) > 0:
        beispiel_obj = train_objects[0]
        print("\nVerfügbare Features in der Beispieldatei:")
        print(list(beispiel_obj.data.columns))
        
        # Für automatisierte Nutzung: Features hier festlegen
        feature_names = [
            'Strom_Spindel', 'Strom_X-Achse', 'Strom_Y-Achse', 'Strom_Z-Achse'
        ]
        
        # Prüfe, ob Features verfügbar sind
        available_features = list(beispiel_obj.data.columns)
        feature_names = [f for f in feature_names if f in available_features]
        
        if not feature_names:
            print("Keine der spezifizierten Features gefunden. Verwende erste 4 verfügbare Features.")
            feature_names = available_features[:4]
        
        print(f"Verwendete Features: {feature_names}")
    else:
        print("Keine Trainingsdaten geladen, Feature-Auswahl nicht möglich.")
        return

    # Windowing-Parameter - Verbesserte Werte
    window_size = 128   # Größeres Fenster für mehr Kontext
    step_size = 32      # Kleinere Schrittweite für mehr Überlappung

    print(f"\nWindowing-Parameter: window_size={window_size}, step_size={step_size}")

    # Fenster erstellen und normalisieren für Training und Test
    print("Erstelle Fenster für Trainingsdaten...")
    X_windows_train, file_idx_train, norm_params_train = create_windows_and_normalize_improved(
        train_objects, feature_names, window_size, step_size
    )
    
    print("Erstelle Fenster für Testdaten...")
    X_windows_test, file_idx_test, norm_params_test = create_windows_and_normalize_improved(
        test_objects, feature_names, window_size, step_size
    )
    
    print(f"Train-Shape: {X_windows_train.shape}, Test-Shape: {X_windows_test.shape}")

    # Für Feedforward-Autoencoder: Fenster zu 2D-Array umformen
    X_ae_train = X_windows_train.reshape(X_windows_train.shape[0], -1)
    X_ae_test = X_windows_test.reshape(X_windows_test.shape[0], -1)

    print(f"Autoencoder Input-Shape: Train={X_ae_train.shape}, Test={X_ae_test.shape}")

    # Training des Autoencoders mit expliziter Epochenzahl
    print("\nStarte Training des verbesserten Feedforward-Autoencoders...")
    n_epochs = 50  # Explizit definiert
    model, training_history = improved_train_autoencoder(
        X_ae_train, 
        n_epochs=n_epochs, 
        batch_size=512, 
        lr=1e-3
    )

    # Anwendung des Autoencoders auf Test- und Trainingsdaten
    print("\nWende Autoencoder auf Daten an...")
    
    # Test Rekonstruktion
    X_ae_test_recon = model.predict(X_ae_test, batch_size=512, verbose=0)
    
    # Train Rekonstruktion
    X_ae_train_recon = model.predict(X_ae_train, batch_size=512, verbose=0)

    # Fenster wieder in [n_windows, window_size, n_features] bringen
    n_features = len(feature_names)
    X_windows_test_recon = X_ae_test_recon.reshape(-1, window_size, n_features)
    X_windows_train_recon = X_ae_train_recon.reshape(-1, window_size, n_features)

    # Zeitreihen pro Datei rekonstruieren (Original und Rekonstruktion)
    print("Rekonstruiere Zeitreihen aus Fenstern...")
    orig_train_series = reconstruct_timeseries_from_windows(
        X_windows_train, file_idx_train, train_objects, window_size, step_size, n_features
    )
    recon_train_series = reconstruct_timeseries_from_windows(
        X_windows_train_recon, file_idx_train, train_objects, window_size, step_size, n_features
    )
    orig_test_series = reconstruct_timeseries_from_windows(
        X_windows_test, file_idx_test, test_objects, window_size, step_size, n_features
    )
    recon_test_series = reconstruct_timeseries_from_windows(
        X_windows_test_recon, file_idx_test, test_objects, window_size, step_size, n_features
    )

    # Pro Datei und Feature: RMS der Rekonstruktionsfehler (Training und Test)
    def rms_feature(orig, recon):
        return np.sqrt(np.mean((orig - recon) ** 2))

    print("Berechne RMS-Fehler...")
    rms_per_file_feature = []
    
    # Training
    for file_idx in orig_train_series:
        for feat_idx, feature_name in enumerate(feature_names):
            rms = rms_feature(orig_train_series[file_idx][:, feat_idx], recon_train_series[file_idx][:, feat_idx])
            rms_per_file_feature.append({
                'split': 'train',
                'file_idx': file_idx,
                'feature': feature_name,
                'rms': rms
            })
    
    # Test
    n_train_files = len(train_objects)
    for file_idx in orig_test_series:
        for feat_idx, feature_name in enumerate(feature_names):
            rms = rms_feature(orig_test_series[file_idx][:, feat_idx], recon_test_series[file_idx][:, feat_idx])
            rms_per_file_feature.append({
                'split': 'test',
                'file_idx': file_idx + n_train_files,  # Testdateien-Index fortlaufend nach Training
                'feature': feature_name,
                'rms': rms
            })
    
    rms_df = pd.DataFrame(rms_per_file_feature)

    # Statistiken ausgeben
    print("\n=== ZUSAMMENFASSUNG ===")
    print(f"Trainierte Epochen: {len(training_history['train_losses'])}")
    print(f"Finale Train Loss: {training_history['train_losses'][-1]:.6f}")
    print(f"Finale Validation Loss: {training_history['val_losses'][-1]:.6f}")
    
    print("\nRMS-Fehler Statistiken:")
    train_rms = rms_df[rms_df['split'] == 'train']['rms']
    test_rms = rms_df[rms_df['split'] == 'test']['rms']
    
    print(f"Training RMS - Mean: {train_rms.mean():.6f}, Std: {train_rms.std():.6f}")
    print(f"Test RMS - Mean: {test_rms.mean():.6f}, Std: {test_rms.std():.6f}")
    
    # Erstelle Ausgabeordner mit Datum und Modellinfo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Bestimme Architektur-String
    input_dim = X_ae_train.shape[1]
    arch_info = model.architecture_info
    arch_string = f"{arch_info['input_dim']}_{arch_info['dim1']}_{arch_info['dim2']}_{arch_info['dim3']}_{arch_info['bottleneck']}"
    
    output_dir = f"D:\\01_Diss\\02_Trainings\\{timestamp}_TF_FFNN_AE_{arch_string}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nErstelle Ausgabeordner: {output_dir}")
    
    # Erstelle Plots
    create_plots(training_history, orig_test_series, recon_test_series, feature_names, rms_df, n_train_files, output_dir)
    
    # 1. Speichere das trainierte Modell (TensorFlow Format)
    model_path = os.path.join(output_dir, "autoencoder_model")
    try:
        model.save(model_path, save_format='tf')
        print(f"TensorFlow Modell gespeichert: {model_path}")
    except Exception as e:
        print(f"Fehler beim Speichern des TF Modells: {e}")
    
    # Speichere auch als .h5 für Kompatibilität
    h5_model_path = os.path.join(output_dir, "autoencoder_model.h5")
    try:
        model.save(h5_model_path)
        print(f"H5 Modell gespeichert: {h5_model_path}")
    except Exception as e:
        print(f"Fehler beim Speichern des H5 Modells: {e}")
    
    # Speichere zusätzliche Metadaten
    metadata = {
        'model_architecture': arch_info,
        'feature_names': feature_names,
        'window_size': window_size,
        'step_size': step_size,
        'n_epochs_trained': len(training_history['train_losses']),
        'final_train_loss': float(training_history['train_losses'][-1]),
        'final_val_loss': float(training_history['val_losses'][-1]),
        'framework': 'tensorflow',
        'tf_version': tf.__version__,
        'training_parameters': {
            'batch_size': 512,
            'learning_rate': 1e-3,
            'optimizer': 'AdamW',
            'weight_decay': 1e-4,
            'early_stopping_patience': 10,
            'lr_reduction_patience': 5,
            'lr_reduction_factor': 0.5
        },
        'data_info': {
            'n_train_files': len(train_objects),
            'n_test_files': len(test_objects),
            'n_train_windows': X_windows_train.shape[0],
            'n_test_windows': X_windows_test.shape[0],
            'normalization': 'z-score'
        }
    }
    
    metadata_path = os.path.join(output_dir, "model_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadaten gespeichert: {metadata_path}")
    
    # 2. Speichere Modell-Informationen als Text
    info_path = os.path.join(output_dir, "model_info.txt")
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write("=== TENSORFLOW AUTOENCODER MODELL INFORMATIONEN ===\n")
        f.write(f"Erstellungsdatum: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n")
        f.write(f"Framework: TensorFlow {tf.__version__}\n")
        f.write(f"Datenordner: {data_folder}\n\n")
        
        f.write("=== DATEN ===\n")
        f.write(f"Anzahl Trainingsdateien: {len(train_objects)}\n")
        f.write(f"Anzahl Testdateien: {len(test_objects)}\n")
        f.write(f"Features: {feature_names}\n")
        f.write(f"Anzahl Features: {len(feature_names)}\n\n")
        
        f.write("=== PREPROCESSING ===\n")
        f.write(f"Window Size: {window_size}\n")
        f.write(f"Step Size: {step_size}\n")
        f.write("Normalisierung: Z-Score pro Datei\n")
        f.write(f"Train Windows: {X_windows_train.shape[0]}\n")
        f.write(f"Test Windows: {X_windows_test.shape[0]}\n\n")
        
        f.write("=== MODELL ARCHITEKTUR ===\n")
        f.write("Typ: Feed-Forward Autoencoder (TensorFlow/Keras)\n")
        f.write(f"Encoder: {input_dim} -> {arch_info['dim1']} -> {arch_info['dim2']} -> {arch_info['dim3']} -> {arch_info['bottleneck']}\n")
        f.write(f"Decoder: {arch_info['bottleneck']} -> {arch_info['dim3']} -> {arch_info['dim2']} -> {arch_info['dim1']} -> {input_dim}\n")
        f.write("Aktivierung: LeakyReLU(0.2)\n")
        f.write("Regularisierung: BatchNorm + Dropout(0.2)\n")
        f.write("Optimizer: AdamW mit Weight Decay 1e-4\n")
        f.write("Loss: MSE\n\n")
        
        f.write("=== TRAINING ===\n")
        f.write(f"Epochen geplant: {n_epochs}\n")
        f.write(f"Epochen trainiert: {len(training_history['train_losses'])}\n")
        f.write(f"Batch Size: 512\n")
        f.write(f"Learning Rate: 1e-3\n")
        f.write(f"Early Stopping: Ja (Patience: 10)\n")
        f.write(f"LR Reduction: Ja (Patience: 5, Factor: 0.5)\n")
        f.write(f"Train/Val Split: 80/20\n\n")
        
        f.write("=== ERGEBNISSE ===\n")
        f.write(f"Finale Train Loss: {training_history['train_losses'][-1]:.6f}\n")
        f.write(f"Finale Validation Loss: {training_history['val_losses'][-1]:.6f}\n")
        f.write(f"Training RMS - Mean: {train_rms.mean():.6f}, Std: {train_rms.std():.6f}\n")
        f.write(f"Test RMS - Mean: {test_rms.mean():.6f}, Std: {test_rms.std():.6f}\n")
        
    print(f"Modell-Info gespeichert: {info_path}")
    
    # 3. Speichere Training History
    history_df = pd.DataFrame({
        'epoch': range(1, len(training_history['train_losses']) + 1),
        'train_loss': training_history['train_losses'],
        'val_loss': training_history['val_losses'],
        'train_mae': training_history['train_mae'],
        'val_mae': training_history['val_mae']
    })
    history_path = os.path.join(output_dir, "training_history.csv")
    history_df.to_csv(history_path, index=False)
    print(f"Training History gespeichert: {history_path}")
    
    # 4. CSV für Excel-freundliches Format (Features in Spalten)
    excel_results = []
    
    # Training data
    for file_idx in orig_train_series:
        file_data = {'split': 'train', 'file_idx': file_idx, 'absolute_file_idx': file_idx}
        for feat_idx, feature_name in enumerate(feature_names):
            rms = rms_feature(orig_train_series[file_idx][:, feat_idx], recon_train_series[file_idx][:, feat_idx])
            file_data[f'RMS_{feature_name}'] = rms
        excel_results.append(file_data)
    
    # Test data  
    for file_idx in orig_test_series:
        file_data = {'split': 'test', 'file_idx': file_idx, 'absolute_file_idx': file_idx + len(train_objects)}
        for feat_idx, feature_name in enumerate(feature_names):
            rms = rms_feature(orig_test_series[file_idx][:, feat_idx], recon_test_series[file_idx][:, feat_idx])
            file_data[f'RMS_{feature_name}'] = rms
        excel_results.append(file_data)
    
    excel_df = pd.DataFrame(excel_results)
    excel_path = os.path.join(output_dir, "reconstruction_errors_excel.csv")
    excel_df.to_csv(excel_path, index=False)
    print(f"Excel-freundliche CSV gespeichert: {excel_path}")
    
    # 5. Originale RMS CSV (für Kompatibilität)
    rms_path = os.path.join(output_dir, "reconstruction_errors_original.csv")
    rms_df.to_csv(rms_path, index=False)
    print(f"Original RMS CSV gespeichert: {rms_path}")
    
    # 6. Speichere Normalisierungsparameter
    norm_info = {
        'train_files': [],
        'test_files': [],
        'feature_names': feature_names,
        'normalization_method': 'z-score'
    }
    
    for i, params in enumerate(norm_params_train):
        norm_info['train_files'].append({
            'file_idx': i,
            'filename': train_files[i] if i < len(train_files) else f"train_{i}",
            'mean': params['mean'].tolist(),
            'std': params['std'].tolist()
        })
    
    for i, params in enumerate(norm_params_test):
        norm_info['test_files'].append({
            'file_idx': i,
            'filename': test_files[i] if i < len(test_files) else f"test_{i}",
            'mean': params['mean'].tolist(),
            'std': params['std'].tolist()
        })
    
    norm_path = os.path.join(output_dir, "normalization_params.json")
    with open(norm_path, 'w') as f:
        json.dump(norm_info, f, indent=2)
    print(f"Normalisierungsparameter gespeichert: {norm_path}")
    
    # 7. Erstelle README für den Ordner
    readme_path = os.path.join(output_dir, "README.txt")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("=== TENSORFLOW AUTOENCODER ORDNER INHALT ===\n\n")
        f.write("autoencoder_model/ - Trainiertes TensorFlow SavedModel Format\n")
        f.write("autoencoder_model.h5 - Trainiertes Keras H5 Format\n")
        f.write("model_metadata.json - Maschinenlesbare Modell-Metadaten (JSON)\n")
        f.write("model_info.txt - Menschenlesbare Modell-Informationen\n")
        f.write("training_history.csv - Loss-Verlauf während Training\n")
        f.write("reconstruction_errors_excel.csv - RMS-Fehler (Excel-Format, Features als Spalten)\n")
        f.write("reconstruction_errors_original.csv - RMS-Fehler (Original-Format)\n")
        f.write("normalization_params.json - Z-Score Parameter pro Datei\n")
        f.write("training_loss.png - Training Loss Plots\n")
        f.write("reconstruction_example.png - Beispiel Rekonstruktion\n")
        f.write("rms_errors.png - RMS Fehler pro Datei\n")
        f.write("README.txt - Diese Datei\n\n")
        
        f.write("=== TENSORFLOW MODELL LADEN ===\n\n")
        f.write("# Python Code zum Laden des SavedModel:\n")
        f.write("import tensorflow as tf\n")
        f.write("model = tf.keras.models.load_model('autoencoder_model')\n\n")
        
        f.write("# Alternativ H5 Format laden:\n")
        f.write("model = tf.keras.models.load_model('autoencoder_model.h5')\n\n")
        
        f.write("# Metadaten laden:\n")
        f.write("import json\n")
        f.write("with open('model_metadata.json', 'r') as f:\n")
        f.write("    metadata = json.load(f)\n\n")
        
        f.write("# Für Inference:\n")
        f.write("# 1. Daten gleich normalisieren wie Training (Z-Score)\n")
        f.write("# 2. In Fenster aufteilen (window_size, step_size aus Metadaten)\n")
        f.write("# 3. Zu 2D umformen: windows.reshape(n_windows, -1)\n")
        f.write("# 4. Modell anwenden: reconstructed = model.predict(data)\n")
        f.write("# 5. Zurück zu 3D: reconstructed.reshape(-1, window_size, n_features)\n")
    
    print(f"README erstellt: {readme_path}")
    print(f"\n=== ALLE DATEIEN GESPEICHERT IN: {output_dir} ===")

if __name__ == "__main__":
    main()