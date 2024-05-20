import tkinter as tk
from tkinter import ttk
import sqlite3
import json
import matplotlib.pyplot as plt
import io
from PIL import Image, ImageTk
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.fftpack import fft
import pywt
from datetime import datetime
from rtlsdr import RtlSdr

# Ścieżka do pliku bazodanowego
DB_FILE_PATH = 'pomiary.db'

# Funkcja do pobierania danych z bazy danych (tylko ostatnia próbka)
def fetch_data_from_database(num_samples=10):
    connection = sqlite3.connect(DB_FILE_PATH)  # Nawiązanie połączenia z bazą danych
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM measurements ORDER BY timestamp DESC LIMIT ?", (num_samples,))
    rows = cursor.fetchall()  # Pobranie ostatnich rekordów z bazy danych
    connection.close()
    return rows

# Funkcja do przetwarzania danych z bazy danych
def process_data_for_graph(data):
    timestamps = [row[1] for row in data]  # Pobranie timestampów z rekordów
    signal_strengths = [json.loads(row[2]) for row in data]  # Pobranie i odczytanie wartości sygnału z rekordów
    return timestamps, signal_strengths

# Funkcja do generowania wykresu sygnału w czasie
def plot_measurements(timestamps, signals):
    plt.figure(figsize=(10, 5))
    plt.plot(signals[-1], label=timestamps[-1])  # Rysowanie tylko jednej próbki sygnału
    plt.title('Signal Strength Over Time')
    plt.xlabel('Sample Number')
    plt.ylabel('Signal Strength (dB)')
    plt.legend(loc='upper right')
    
    # Zapis wykresu do obrazu
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    
    return img

# Funkcja do generowania wykresu korelacji
def plot_correlation(signals):
    if len(signals) < 2:
        return None
    
    correlation = np.correlate(signals[-1], signals[-2], mode='full')  # Obliczenie korelacji sygnałów
    
    plt.figure(figsize=(10, 5))
    plt.plot(correlation)
    plt.title('Correlation of Signals')
    plt.xlabel('Lag')
    plt.ylabel('Correlation')
    
    # Zapis wykresu do obrazu
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    
    return img

# Funkcja do generowania wykresu FFT na oryginalnych wartościach
def plot_fft_original(signals):
    num_samples = len(signals[0])  # Liczba próbek sygnału
    
    sampling_freq = 2 * num_samples  # Próbkowanie zgodne z twierdzeniem Nyquista-Shannona
    freq = np.fft.fftfreq(num_samples, d=1/sampling_freq)
    
    fft_result = np.fft.fft(signals[0])  # Analiza FFT na pierwszej próbce
    
    plt.figure(figsize=(10, 5))
    plt.plot(freq, np.abs(fft_result))
    plt.title('Fast Fourier Transform (FFT) - Original Values')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    
    # Zapis wykresu do obrazu
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    
    return img

# Funkcja do transformacji falkowej
def wavelet_transform(signals):
    coeffs = pywt.wavedec(signals[-1], 'db1', level=4)  # Przeprowadzenie transformacji falkowej
    plt.figure(figsize=(10, 5))
    for i in range(len(coeffs)):
        plt.plot(coeffs[i], label='Level {}'.format(i))  # Rysowanie współczynników falkowych na różnych poziomach
    plt.title('Wavelet Transform')
    plt.xlabel('Coefficient Index')
    plt.ylabel('Amplitude')
    plt.legend()
    
    # Zapis wykresu do obrazu
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    
    return img

# Funkcja do transformacji Laplace'a
def laplace_transform(signals):
    laplace_result = np.diff(signals[-1])  # Przeprowadzenie transformacji Laplace'a (różniczkowanie)
    
    plt.figure(figsize=(10, 5))
    plt.plot(laplace_result)
    plt.title('Laplace Transform')
    plt.xlabel('Sample Number')
    plt.ylabel('Transformed Signal')
    
    # Zapis wykresu do obrazu
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    
    return img

# Funkcja do aproksymacji krzywą
def curve_approximation(signals):
    smoothed_signal = savgol_filter(signals[-1], 51, 3)  # Wygładzanie sygnału za pomocą filtru Savitzky-Golaya
    peaks, _ = find_peaks(smoothed_signal, distance=20)  # Znajdowanie szczytów w wygładzonym sygnale
    
    plt.figure(figsize=(10, 5))
    plt.plot(signals[-1], label='Original Signal')
    plt.plot(peaks, smoothed_signal[peaks], 'x', label='Peaks')  # Rysowanie wykresu sygnału z zaznaczonymi szczytami
    plt.title('Curve Approximation')
    plt.xlabel('Sample Number')
    plt.ylabel('Signal Strength (dB)')
    plt.legend()
    
    # Zapis wykresu do obrazu
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    
    return img

# Funkcja do zapisywania danych do bazy danych
def save_data_to_database(timestamp, signal_strength):
    connection = sqlite3.connect(DB_FILE_PATH)  # Nawiązanie połączenia z bazą danych
    cursor = connection.cursor()
    
    # Tworzenie tabeli, jeśli nie istnieje
    cursor.execute('''CREATE TABLE IF NOT EXISTS measurements
                      (id INTEGER PRIMARY KEY, timestamp TEXT, signal_strength TEXT)''')
    
    # Wstawianie danych do tabeli
    cursor.execute("INSERT INTO measurements (timestamp, signal_strength) VALUES (?, ?)",
                   (timestamp, json.dumps(signal_strength)))
    
    connection.commit()
    connection.close()

# Funkcja do zbierania danych z SDR-RTL
def collect_sdr_data(frequency=433e6, sample_rate=2.048e6, num_samples=1024):
    sdr = RtlSdr()  # Inicjalizacja SDR-RTL
    
    # Konfiguracja SDR
    sdr.sample_rate = sample_rate
    sdr.center_freq = frequency
    sdr.gain = 'auto'
    
    # Odbieranie sygnału
    samples = sdr.read_samples(num_samples)
    sdr.close()
    
    # Obliczanie mocy sygnału
    signal_strength = 10 * np.log10(np.abs(samples)**2)
    
    # Pobranie aktualnego czasu
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Zapis danych do bazy
    save_data_to_database(timestamp, signal_strength.tolist())

# Klasa aplikacji GUI
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Wykresy")
        
        # Etykieta i obszar dla wykresu sygnału w czasie
        self.graph_label = ttk.Label(self.root, text="Wykres: Sygnał w czasie (Ostatnia próbka)")
        self.graph_label.pack()
        self.graph_canvas = tk.Canvas(self.root, width=600, height=300)
        self.graph_canvas.pack()
        
        # Etykieta i obszar dla wykresu korelacji
        self.correlation_label = ttk.Label(self.root, text="Wykres: Korelacja sygnałów")
        self.correlation_label.pack()
        self.correlation_canvas = tk.Canvas(self.root, width=600, height=300)
        self.correlation_canvas.pack()
        
        # Etykieta i obszar dla wykresu FFT
        self.fft_label = ttk.Label(self.root, text="Wykres: Szybka Transformata Fouriera (FFT) - Oryginalne wartości")
        self.fft_label.pack()
        self.fft_canvas = tk.Canvas(self.root, width=600, height=300)
        self.fft_canvas.pack()

        # Etykieta i obszar dla wykresu transformacji falkowej
        self.wavelet_label = ttk.Label(self.root, text="Wykres: Transformata Falkowa")
        self.wavelet_label.pack()
        self.wavelet_canvas = tk.Canvas(self.root, width=600, height=300)
        self.wavelet_canvas.pack()
        
        # Etykieta i obszar dla wykresu transformacji Laplace'a
        self.laplace_label = ttk.Label(self.root, text="Wykres: Transformata Laplace'a")
        self.laplace_label.pack()
        self.laplace_canvas = tk.Canvas(self.root, width=600, height=300)
        self.laplace_canvas.pack()
        
        # Etykieta i obszar dla wykresu aproksymacji krzywą
        self.curve_label = ttk.Label(self.root, text="Wykres: Aproksymacja Krzywą")
        self.curve_label.pack()
        self.curve_canvas = tk.Canvas(self.root, width=600, height=300)
        self.curve_canvas.pack()
        
        self.update_graphs()  # Pierwsze wywołanie funkcji aktualizacji wykresów
    
    def update_graphs(self):
        collect_sdr_data()  # Zbieranie danych z SDR-RTL
        data = fetch_data_from_database(10)  # Pobranie danych z bazy
        timestamps, signal_strengths = process_data_for_graph(data)  # Przetwarzanie danych do wykresów
        
        # Generowanie i wyświetlanie wykresu sygnału w czasie
        graph_img = plot_measurements(timestamps, signal_strengths)
        self.graph_img_tk = ImageTk.PhotoImage(Image.open(graph_img))
        self.graph_canvas.create_image(0, 0, anchor=tk.NW, image=self.graph_img_tk)
        
        # Generowanie i wyświetlanie wykresu korelacji
        correlation_img = plot_correlation(signal_strengths)
        if correlation_img:
            self.correlation_img_tk = ImageTk.PhotoImage(Image.open(correlation_img))
            self.correlation_canvas.create_image(0, 0, anchor=tk.NW, image=self.correlation_img_tk)
        
        # Generowanie i wyświetlanie wykresu FFT
        fft_img = plot_fft_original(signal_strengths)
        self.fft_img_tk = ImageTk.PhotoImage(Image.open(fft_img))
        self.fft_canvas.create_image(0, 0, anchor=tk.NW, image=self.fft_img_tk)

        # Generowanie i wyświetlanie wykresu transformacji falkowej
        wavelet_img = wavelet_transform(signal_strengths)
        self.wavelet_img_tk = ImageTk.PhotoImage(Image.open(wavelet_img))
        self.wavelet_canvas.create_image(0, 0, anchor=tk.NW, image=self.wavelet_img_tk)
        
        # Generowanie i wyświetlanie wykresu transformacji Laplace'a
        laplace_img = laplace_transform(signal_strengths)
        self.laplace_img_tk = ImageTk.PhotoImage(Image.open(laplace_img))
        self.laplace_canvas.create_image(0, 0, anchor=tk.NW, image=self.laplace_img_tk)
        
        # Generowanie i wyświetlanie wykresu aproksymacji krzywą
        curve_img = curve_approximation(signal_strengths)
        self.curve_img_tk = ImageTk.PhotoImage(Image.open(curve_img))
        self.curve_canvas.create_image(0, 0, anchor=tk.NW, image=self.curve_img_tk)
        
        self.root.after(5000, self.update_graphs)  # Aktualizacja co 5 sekund

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
