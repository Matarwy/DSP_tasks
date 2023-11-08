import numpy as np
import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import math


def when_changed():
    st.session_state.Addition_uploaded_files = []
    st.session_state.Subtraction_Signal1 = None
    st.session_state.Subtraction_Signal2 = None
    st.session_state.Multiplication_Signal = None
    st.session_state.Multiplier = 1
    st.session_state.Squaring_Signal = None
    st.session_state.Shifting_Signal = None
    st.session_state.Shifter = 0
    st.session_state.Normalization_Signal = None
    st.session_state.Normalizer_min = -1
    st.session_state.Normalizer_max = 1
    st.session_state.Accumulation_Signal = None


def plot_chart(signal):
    chart = make_subplots(rows=1, cols=1)
    chart.add_trace(go.Scatter(y=signal[:, 1], x=signal[:, 0], mode="lines", name="continuous"), row=1, col=1)
    chart.add_trace(go.Scatter(y=signal[:, 1], x=signal[:, 0], mode="markers", name="discrete"), row=1, col=1)
    chart.update_xaxes(title_text='Index')
    chart.update_yaxes(title_text='SampleAmp')
    chart.update_layout(autosize=True)
    st.plotly_chart(chart, use_container_width=True)


def plot_bar(signal):
    chart = make_subplots(rows=1, cols=1)
    chart.add_trace(go.Bar(
        x=st.session_state.freqdomain_signal[:, 0],
        y=st.session_state.freqdomain_signal[:, 1],
        marker=dict(color='red'), width=0.2,
        text=['{:.1f}'.format(x[1]) for x in st.session_state.freqdomain_signal],
        textposition='outside'
    ))
    chart.update_xaxes(title_text='frequency')
    chart.update_yaxes(title_text='amplitude')
    chart.update_layout(autosize=True)
    st.plotly_chart(chart, use_container_width=True)
    chart = make_subplots(rows=1, cols=1)
    chart.add_trace(go.Bar(
        x=st.session_state.freqdomain_signal[:, 0],
        y=st.session_state.freqdomain_signal[:, 2],
        marker=dict(color='green'), width=0.2,
        text=['{:.1f}'.format(x[2]) for x in st.session_state.freqdomain_signal],
        textposition='outside'
    ))
    chart.update_xaxes(title_text='frequency')
    chart.update_yaxes(title_text='phase shift')
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
        signal.append([n, sum])
    Signal = np.array(signal)
    return Signal


def Quantization(signal_input, num_of_levels):
    quntized = []
    errors = []
    ranges = []
    midpoints = []
    encoded = []
    interval = []
    SignalType, IsPeriodic, N1, signal = read_file(signal_input)
    max_value = max(signal[:, 1])
    min_value = min(signal[:, 1])
    delta = (max_value - min_value) / num_of_levels
    sum = min_value
    for i in range(0, num_of_levels):
        max_range = sum + delta
        ranges.append([round(sum, 5), round(max_range, 5)])
        midpoints.append(round((sum + max_range)/2, 5))
        sum = max_range

    for sample in signal:
        for i in range(0, len(ranges)):
            if ranges[i][0] <= sample[1] <= ranges[i][1]:
                quntized.append([sample[0], midpoints[i]])
                interval.append(i + 1)
                encoded.append(bin(i)[2:].zfill(int(np.log2(num_of_levels))))
                errors.append([sample[0], round(midpoints[i] - sample[1], 5)])
                break
    interval = np.array(interval)
    encoded = np.array(encoded)
    quntized = np.array(quntized)
    errors = np.array(errors)
    return interval, encoded, quntized, errors


def dft(SignalType, N1, signal, fs):
    if SignalType == 0:
        st.session_state.timedomain_signal = signal
        freqdomain_signal = []
        omega = (2 * np.pi * fs) / N1
        for k in range(0, N1):
            real = 0
            imagine = 0
            for n in range(0, N1):
                if k == 0 and n == 0:
                    real += signal[n, 1]
                else:
                    e_power = ((2 * math.pi * k * n) / N1)
                    real += (signal[n, 1] * np.cos(e_power))
                    imagine += (signal[n, 1] * np.sin(e_power) * -1)
            freqdomain_signal.append([
                float((k+1) * omega),
                float(math.sqrt((real ** 2) + (imagine ** 2))),
                float(math.atan2(imagine, real))
            ])
            st.session_state.freqdomain_signal = np.array(freqdomain_signal)
    if SignalType == 1:
        st.session_state.freqdomain_signal = signal
        timedomain_signal = []
        for n in range(0, N1):
            realsum = 0
            imaginsum = 0
            for k in range(0, N1):
                real = signal[k, 1] * np.cos(signal[k, 2])
                imagine = signal[k, 1] * np.sin(signal[k, 2])
                e_power = (2 * math.pi * n * k) / N1
                realsum += round((real * np.cos(e_power)) - (imagine * np.sin(e_power)), 2)
                imaginsum += round((imagine * np.cos(e_power)) + (real * np.sin(e_power)), 2)
            if imaginsum != 0:
                print("Error: while calculating imagine part in idft")
            sample = realsum / N1
            timedomain_signal.append([
                n,
                float(sample)
            ])
            st.session_state.timedomain_signal = np.array(timedomain_signal)


