# 18.07.2025 - RR
# Dieses Skript liest alle TransKI Daten aus einem Ordner, erstellt Zeitfenster für eine beliebige Anzahl von Features und normalisiert diese.
# Anschließend werden verschiedene ML Modelle trainiert (CNN - AE, FFNN - AE, XGBoost, Random Forest) und die Ergebnisse in einer CSV-Datei gespeichert.

import pandas as pd
import numpy as np
from helper_functions import ifw_data
import os
import torch
import torch.nn as nn
import torch.optim as optim
import re
import matplotlib.pyplot as plt

def create_windows_and_normalize(data_objects, feature_names, window_size, step_size):
    """
    Erzeugt normalisierte Zeitfenster für die angegebenen Features aus allen Datenobjekten.
    Normalisierung erfolgt pro Datei (nicht pro Fenster).
    Gibt zusätzlich eine Liste zurück, die für jedes Fenster den Index der Ursprungsdatei enthält.
    """
    windows = []
    file_indices = []
    for file_idx, obj in enumerate(data_objects):
        df = obj.data[feature_names].copy()
        arr = df.values
        n = arr.shape[0]
        # Min-Max Normalisierung pro Datei
        min_ = arr.min(axis=0)
        max_ = arr.max(axis=0)
        denom = (max_ - min_)
        denom[denom == 0] = 1  # Division durch 0 vermeiden
        arr_norm = (arr - min_) / denom
        # Sliding window
        for start in range(0, n - window_size + 1, step_size):
            window = arr_norm[start:start+window_size]
            windows.append(window)
            file_indices.append(file_idx)
    return np.array(windows), np.array(file_indices)

class FeedForwardAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

def train_autoencoder(X, n_epochs=60, batch_size=512, lr=1e-3):
    print(f"Starte mit n_epochs={n_epochs}")  # Debug-Ausgabe
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = FeedForwardAutoencoder(X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{n_epochs} - Loss: {epoch_loss:.6f}")
    return model, losses

def reconstruct_timeseries_from_windows(windows, file_indices, n_files, window_size, step_size, n_features):
    """
    Rekonstruiert die Zeitreihen pro Datei und Feature aus überlappenden Fenstern (mittelt Überlappungen).
    Gibt ein Dict {file_idx: np.ndarray [n_samples, n_features]} zurück.
    """
    # Zähle, wie viele Fenster pro Datei
    file_n_windows = {i: np.sum(file_indices == i) for i in range(n_files)}
    file_n_samples = {i: (file_n_windows[i] - 1) * step_size + window_size if file_n_windows[i] > 0 else 0 for i in range(n_files)}

    # Initialisiere Arrays für Rekonstruktion und Zähler für Mittelung
    timeseries = {i: np.zeros((file_n_samples[i], n_features)) for i in range(n_files)}
    counts = {i: np.zeros((file_n_samples[i], n_features)) for i in range(n_files)}

    # Lokaler Fensterzähler pro Datei
    local_win_idx = {i: 0 for i in range(n_files)}

    for global_win_idx, file_idx in enumerate(file_indices):
        if file_n_samples[file_idx] == 0:
            continue
        local_idx = local_win_idx[file_idx]
        start = local_idx * step_size
        end = start + window_size
        if end > file_n_samples[file_idx]:
            end = file_n_samples[file_idx]
            window = windows[global_win_idx][:end-start]
        else:
            window = windows[global_win_idx]
        timeseries[file_idx][start:end] += window
        counts[file_idx][start:end] += 1
        local_win_idx[file_idx] += 1

    # Mittelung der Überlappungen
    for i in range(n_files):
        mask = counts[i] > 0
        timeseries[i][mask] /= counts[i][mask]
    return timeseries

def main():
    # Verzeichnis mit TransKI-Daten
    data_folder = r'D:\\01_Diss\\01_Versuchsdaten\\01_TransKI\\01_Fraesen_Stand\\IfW\\Grob\\Toolox33\\Toolox33_Standzeit_IfW_eID1156'  # ggf. anpassen

    # Alle Dateien im Verzeichnis auflisten
    files = [f for f in os.listdir(data_folder) if f.endswith('.txt') or "Maschinendaten" in f]

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

    # Liste für ifw_data-Objekte
    train_objects = []
    test_objects = []

    for file in train_files:
        try:
            obj = ifw_data(data_folder, file, 'TransKI')
            train_objects.append(obj)
        except Exception as e:
            print(f"Fehler beim Laden von {file}: {e}")

    for file in test_files:
        try:
            obj = ifw_data(data_folder, file, 'TransKI')
            test_objects.append(obj)
        except Exception as e:
            print(f"Fehler beim Laden von {file}: {e}")

    print(f"{len(train_objects)} Trainingsdateien, {len(test_objects)} Testdateien erfolgreich geladen.")

    # --- Feature-Abfrage für eine Beispiel-Datei ---
    if len(train_objects) > 0:
        beispiel_obj = train_objects[0]
        print("Verfügbare Features in der Beispieldatei:")
        print(list(beispiel_obj.data.columns))
        # Optional: Hier könnten Sie eine Benutzereingabe einbauen, z.B.:
        # feature_names = input("Geben Sie die gewünschten Features kommasepariert ein: ").split(",")
        # feature_names = [f.strip() for f in feature_names]
        # Für automatisierte Nutzung: Features hier festlegen
        feature_names = [
            # Beispiel: 'AE_Spindel', 'AE_X', 'AE_Y', 'AE_Z'
            # Passe die Namen an die tatsächlichen Spaltennamen an!
            'Strom_Spindel', 'Strom_X-Achse', 'Strom_Y-Achse', 'Strom_Z-Achse'
        ]
    else:
        print("Keine Trainingsdaten geladen, Feature-Auswahl nicht möglich.")
        return

    # Windowing-Parameter
    window_size = 64   # z.B. 256 Datenpunkte pro Fenster
    step_size = 16      # Schrittweite (z.B. 64 Datenpunkte)

    # Fenster erstellen und normalisieren für Training und Test
    X_windows_train, file_idx_train = create_windows_and_normalize(train_objects, feature_names, window_size, step_size)
    X_windows_test, file_idx_test = create_windows_and_normalize(test_objects, feature_names, window_size, step_size)
    print(f"Train-Shape: {X_windows_train.shape}, Test-Shape: {X_windows_test.shape}")

    # Für Feedforward-Autoencoder: Fenster zu 2D-Array umformen
    X_ae_train = X_windows_train.reshape(X_windows_train.shape[0], -1)
    X_ae_test = X_windows_test.reshape(X_windows_test.shape[0], -1)

    # Training des Autoencoders
    print("Starte Training des Feedforward-Autoencoders...")
    # Hier wird train_autoencoder ohne explizite n_epochs aufgerufen:
    model, losses = train_autoencoder(X_ae_train, n_epochs=60, batch_size=512, lr=1e-3)
    # Das bedeutet, dass der Default-Wert aus der Funktionsdefinition verwendet wird:
    # def train_autoencoder(X, n_epochs=120, batch_size=256, lr=1e-3):

    # Visualisierung des Trainings-Loss
    plt.figure(figsize=(8, 5))
    plt.plot(losses, marker='o')
    plt.xlabel('Epoche')
    plt.ylabel('Loss')
    plt.title('Trainings-Loss pro Epoche')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Anwendung des Autoencoders auf Testdaten
    model.eval()
    with torch.no_grad():
        X_ae_test_tensor = torch.tensor(X_ae_test, dtype=torch.float32)
        X_ae_test_recon = model(X_ae_test_tensor).cpu().numpy()
        X_ae_train_tensor = torch.tensor(X_ae_train, dtype=torch.float32)
        X_ae_train_recon = model(X_ae_train_tensor).cpu().numpy()

    # Fenster wieder in [n_windows, window_size, n_features] bringen
    n_features = len(feature_names)
    X_windows_test_recon = X_ae_test_recon.reshape(-1, window_size, n_features)
    X_windows_train_recon = X_ae_train_recon.reshape(-1, window_size, n_features)

    # Zeitreihen pro Datei rekonstruieren (Original und Rekonstruktion)
    n_train_files = len(train_objects)
    n_test_files = len(test_objects)
    orig_train_series = reconstruct_timeseries_from_windows(X_windows_train, file_idx_train, n_train_files, window_size, step_size, n_features)
    recon_train_series = reconstruct_timeseries_from_windows(X_windows_train_recon, file_idx_train, n_train_files, window_size, step_size, n_features)
    orig_test_series = reconstruct_timeseries_from_windows(X_windows_test, file_idx_test, n_test_files, window_size, step_size, n_features)
    recon_test_series = reconstruct_timeseries_from_windows(X_windows_test_recon, file_idx_test, n_test_files, window_size, step_size, n_features)

    # Beispielplot: Original vs. Rekonstruktion für eine Testdatei und ein Feature
    test_file_idx = 0
    feature_idx = 0  # z.B. 0 für 'Strom_Spindel', 1 für 'Strom_X-Achse', ...
    if test_file_idx in orig_test_series:
        orig = orig_test_series[test_file_idx]
        recon = recon_test_series[test_file_idx]
        if feature_idx < orig.shape[1]:
            plt.figure(figsize=(12, 4))
            plt.plot(orig[:, feature_idx], label='Original', alpha=0.7)
            plt.plot(recon[:, feature_idx], label='Rekonstruktion', alpha=0.7)
            plt.title(f'Original vs. Rekonstruktion (Testdatei {test_file_idx+1}, Feature: {feature_names[feature_idx]})')
            plt.xlabel('Zeitindex')
            plt.ylabel('Normierter Wert')
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            print(f"feature_idx {feature_idx} liegt außerhalb der verfügbaren Features ({orig.shape[1]}).")
    else:
        print(f"Testdatei-Index {test_file_idx} nicht in orig_test_series.")

    # Pro Datei und Feature: RMS der Rekonstruktionsfehler (Training und Test)
    def rms_feature(orig, recon):
        return np.sqrt(np.mean((orig - recon) ** 2))

    rms_per_file_feature = []
    # Training
    for file_idx in orig_train_series:
        for feat_idx in range(n_features):
            rms = rms_feature(orig_train_series[file_idx][:, feat_idx], recon_train_series[file_idx][:, feat_idx])
            rms_per_file_feature.append({
                'split': 'train',
                'file_idx': file_idx,
                'feature': feature_names[feat_idx],
                'rms': rms
            })
    # Test
    for file_idx in orig_test_series:
        for feat_idx in range(n_features):
            rms = rms_feature(orig_test_series[file_idx][:, feat_idx], recon_test_series[file_idx][:, feat_idx])
            rms_per_file_feature.append({
                'split': 'test',
                'file_idx': file_idx + n_train_files,  # Testdateien-Index fortlaufend nach Training
                'feature': feature_names[feat_idx],
                'rms': rms
            })
    rms_df = pd.DataFrame(rms_per_file_feature)

    # Plot: RMS pro Datei (x-Achse: Datei-Index fortlaufend, Linien: Features)
    plt.figure(figsize=(12, 5))
    for feat in feature_names:
        subset = rms_df[rms_df['feature'] == feat].sort_values('file_idx')
        plt.plot(subset['file_idx'], subset['rms'], marker='o', linestyle='-', label=feat)
    # Linie für Train/Test-Split
    plt.axvline(n_train_files - 0.5, color='red', linestyle='--', label='Train/Test-Grenze')
    plt.xlabel('Datei-Index (Train + Test)')
    plt.ylabel('RMS Fehler')
    plt.title('RMS Rekonstruktionsfehler pro Datei (pro Feature)')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


