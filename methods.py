import numpy as np
import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def plot_chart(signal):
    chart = make_subplots(rows=1, cols=1)
    chart.add_trace(go.Scatter(y=signal[:, 1], x=signal[:, 0], mode="lines", name="continuous"), row=1, col=1)
    chart.add_trace(go.Scatter(y=signal[:, 1], x=signal[:, 0], mode="markers", name="discrete"), row=1, col=1)
    chart.update_xaxes(title_text='Index')
    chart.update_yaxes(title_text='SampleAmp')
    chart.update_layout(autosize=True)
    st.plotly_chart(chart, use_container_width=True)


def read_file(file):
    try:
        data = file.readlines()
        SignalType = int(data[0].strip())
        IsPeriodic = int(data[1].strip())
        N1 = int(data[2].strip())
        rows = data[3:]
        signal = []
        if SignalType == 0:
            for row in rows:
                values = row.split()
                Index = float(values[0].strip())
                SampleAmp = float(values[1].strip())
                signal.append([Index, SampleAmp])
        if SignalType == 1:
            for row in rows:
                values = row.split()
                Freq = float(values[0].strip())
                Amp = float(values[1].strip())
                PhaseShift = float(values[2].strip())
                signal.append([Freq, Amp, PhaseShift])
        Signal = np.array(signal)
        return SignalType, IsPeriodic, N1, Signal
    except:
        print("Read File Error")


def generate_signal(type, amp, phaseshift, frequency, samplingfrq):
    signal = []
    N = int(samplingfrq)
    if type == 'sin':
        for n in range(0, N):
            SampleAmp = amp * np.sin((2*np.pi*(frequency/samplingfrq)*n) + phaseshift)
            signal.append([n, SampleAmp])
    if type == 'cos':
        for n in range(0, N):
            SampleAmp = amp * np.cos((2*np.pi*(frequency/samplingfrq)*n) + phaseshift)
            signal.append([n, SampleAmp])
    Signal = np.array(signal)
    return Signal, N


def Addition(Addition_uploaded_files):
    signal = []
    Signals = []
    for s in Addition_uploaded_files:
        SignalType, IsPeriodic, N1, Signal = read_file(s)
        Signals.append(Signal[:, 1])
    output_signal_size = max(len(s) for s in Signals)
    index = 0
    for n in range(0, output_signal_size):
        sum = 0
        for s in Signals:
            if len(s) > index:
                sum += s[n]
        signal.append([n, sum])
    Signal = np.array(signal)
    return Signal


def Subtraction(Signal1, Signal2):
    signal = []
    SignalType1, IsPeriodic1, N1, signal1 = read_file(Signal1)
    SignalType2, IsPeriodic2, N2, signal2 = read_file(Signal2)
    for n in range(0, len(signal1)):
        sample = signal1[n][1] - signal2[n][1]
        if sample < 0:
            sample *= -1
        signal.append([n, sample])
    Signal = np.array(signal)
    return Signal


def Multiplication(Signalm, multiplyer):
    signal = []
    SignalType, IsPeriodic, N1, signalm = read_file(Signalm)
    for n in range(0, len(signalm)):
        signal.append([n, (multiplyer * signalm[n][1])])
    Signal = np.array(signal)
    return Signal


def Squaring(Signals):
    signal = []
    SignalType, IsPeriodic, N1, signalm = read_file(Signals)
    for n in range(0, len(signalm)):
        signal.append([n, (signalm[n][1] * signalm[n, 1])])
    Signal = np.array(signal)
    return Signal


def Shifting(Signals, shifter):
    signal = []
    SignalType, IsPeriodic, N1, signalm = read_file(Signals)
    for n in range(0, len(signalm)):
        signal.append([signalm[n][0] - shifter, signalm[n, 1]])
    Signal = np.array(signal)
    return Signal


def Normalization(Signaln, maxn, minn):
    signal = []
    SignalType, IsPeriodic, N1, signalm = read_file(Signaln)
    normalized_signal = minn + (maxn - minn) * ((signalm[:, 1] - np.min(signalm[:, 1])) / (np.max(signalm[:, 1]) - np.min(signalm[:, 1])))
    for n in range(0, len(signalm)):
        signal.append([n, normalized_signal[n]])
    Signal = np.array(signal)
    return Signal


def Accumulation(SignalA):
    signal = []
    SignalType, IsPeriodic, N1, signalm = read_file(SignalA)
    for n in range(0, len(signalm)):
        sum = 0.0
        for s in range(0, (n +1)):
            sum += signalm[s, 1]
        print(sum)
        signal.append([n, sum])
    Signal = np.array(signal)
    return Signal
