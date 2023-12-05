import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import sqlite3
import json
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

app = dash.Dash(__name__)

# Ścieżka do pliku bazodanowego
DB_FILE_PATH = '/home/ddosser/Desktop/baza_danych/pomiary.db'

# Funkcja do pobierania danych z bazy danych (tylko ostatnia próbka)
def fetch_data_from_database(num_samples=10):
    connection = sqlite3.connect(DB_FILE_PATH)
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM measurements ORDER BY timestamp DESC LIMIT ?", (num_samples,))
    rows = cursor.fetchall()
    connection.close()
    return rows

# Funkcja do przetwarzania danych z bazy danych
def process_data_for_graph(data):
    timestamps = [row[1] for row in data]  # Pobranie timestampów z rekordów
    signal_strengths = [json.loads(row[2]) for row in data]  # Pobranie signal_strength z rekordów
    return timestamps, signal_strengths

# Funkcja do generowania wykresu sygnału w czasie
def plot_measurements(timestamps, signals):
    plt.figure(figsize=(10, 5))
    plt.plot(signals[-1], label=timestamps[-1])  # Rysowanie tylko jednej próbki sygnału
    plt.title('Signal Strength Over Time')
    plt.xlabel('Sample Number')
    plt.ylabel('Signal Strength (dB)')
    plt.legend(loc='upper right')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.read()).decode())

# Funkcja do generowania wykresu korelacji
def plot_correlation(signals):
    correlation = np.correlate(signals[-1], signals[-2], mode='full')
    
    plt.figure(figsize=(10, 5))
    plt.plot(correlation)
    plt.title('Correlation of Signals')
    plt.xlabel('Lag')
    plt.ylabel('Correlation')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.read()).decode())

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
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.read()).decode())

# Wykres
app.layout = html.Div([
    html.H1("Wykresy"),
    html.Div([
        html.H2("Wykres: Sygnał w czasie (Ostatnia próbka)"),
        html.Img(id='graph-image'),
    ]),
    html.Div([
        html.H2("Wykres: Korelacja sygnałów"),
        html.Img(id='correlation-image'),
    ]),
    html.Div([
        html.H2("Wykres: Szybka Transformata Fouriera (FFT) - Oryginalne wartości"),
        html.Img(id='fft-original-image'),
    ]),
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # Przykładowy interwał, co 5 sekund
        n_intervals=0
    )
])

@app.callback(
    Output('graph-image', 'src'),
    Output('correlation-image', 'src'),
    Output('fft-original-image', 'src'),
    Input('interval-component', 'n_intervals')
)
def update_graph(n):
    data = fetch_data_from_database(10)
    timestamps, signal_strengths = process_data_for_graph(data)
    
    graph_base64 = plot_measurements(timestamps, signal_strengths)
    correlation_base64 = plot_correlation(signal_strengths)
    fft_original_base64 = plot_fft_original(signal_strengths)
    return graph_base64, correlation_base64, fft_original_base64

if __name__ == '__main__':
    app.run_server(debug=True, port=8060)
