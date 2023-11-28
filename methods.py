import numpy as np
import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import math
from os import path
from comparesignal2 import SignalSamplesAreEqual as ssae
from ConvTest import *


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
                if len(values) == 2:
                    Amp = float(values[0].strip())
                    PhaseShift = float(values[1].strip())
                    signal.append([Amp, PhaseShift])
                elif len(values) == 3:
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


def dft(SignalType, N1, signal, fs, operation="dft", dc=False):
    if SignalType == 0:
        if dc:
            meanofsignal = np.sum(signal[:, 1]) / len(signal)
            for sam in signal:
                sam[1] -= meanofsignal
            if st.session_state.dft_file.name == "DC_component_input.txt":
                ssae(
                    path.relpath("signals/task5/Remove DC component/DC_component_output.txt"),
                    signal[:, 1]
                )
        st.session_state.timedomain_signal = signal
        freqdomain_signal = []
        omega = (2 * np.pi * fs) / N1
        for k in range(0, N1):
            real = 0
            imagine = 0
            if operation == "dft":
                for n in range(0, N1):
                    if k == 0 and n == 0:
                        real += signal[n, 1]
                    else:
                        e_power = ((2 * math.pi * k * n) / N1)
                        real += (signal[n, 1] * np.cos(e_power))
                        imagine += (signal[n, 1] * np.sin(e_power) * -1)
            elif operation == "dct":
                for n in range(0, N1):
                    real += signal[n, 1] * np.cos((math.pi / (4 * N1)) * ((2 * n) - 1) * ((2 * k) - 1))
                real = math.sqrt(2 / N1) * real

            freqdomain_signal.append([
                float((k+1) * omega),
                float(math.sqrt((real ** 2) + (imagine ** 2))),
                float(math.atan2(imagine, real))
            ])
        if dc:
            if len(freqdomain_signal[0]) == 2:
                freqdomain_signal[0][0] = 0
                freqdomain_signal[0][1] = 0
            if len(freqdomain_signal[0]) == 3:
                freqdomain_signal[0][1] = 0
                freqdomain_signal[0][2] = 0
        if st.session_state.dft_file.name == "DCT_input.txt":
            testSamples = []
            for samp in freqdomain_signal:
                testSamples.append(samp[1])
            ssae(
                path.relpath("signals/task5/DCT/DCT_output.txt"),
                testSamples
            )
        st.session_state.freqdomain_signal = np.array(freqdomain_signal)
    if SignalType == 1:
        if dc:
            if len(signal[0]) == 2:
                signal[0][0] = 0
                signal[0][1] = 0
            if len(signal[0]) == 3:
                signal[0][1] = 0
                signal[0][2] = 0
        st.session_state.freqdomain_signal = signal
        timedomain_signal = []
        if operation == "dft":
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


def timedomain_smoothing(wz=3):
    SignalType, IsPeriodic, N1, signalm = read_file(st.session_state.timedomain)
    signal = []
    for i in range(0, (len(signalm) - wz + 1)):
        averagemean = 0
        for j in range(i, i + wz):
            averagemean += signalm[j, 1]
        averagemean /= wz
        signal.append([i, averagemean])
    Signal = np.array(signal)
    st.header("Smoothing")
    plot_chart(Signal)
    if st.session_state.timedomain.name == "Signal1.txt" and wz == 3:
        ssae(path.relpath("signals/lab6/Moving Average/OutMovAvgTest1.txt"), Signal[:, 1])
    if st.session_state.timedomain.name == "Signal2.txt" and wz == 5:
        ssae(path.relpath("signals/lab6/Moving Average/OutMovAvgTest2.txt"), Signal[:, 1])


def timedomain_delaying_advance(k):
    SignalType, IsPeriodic, N1, signalm = read_file(st.session_state.timedomain)
    signal = []
    for sample in signalm:
        signal.append([sample[0] + k, sample[1]])
    Signal = np.array(signal)
    return Signal


def timedomain_folding(signalm):
    signal = []
    for i in range(0, len(signalm)):
        signal.append([signalm[i, 0], signalm[(len(signalm) - i - 1), 1]])
    Signal = np.array(signal)
    return Signal


def timedomain_convolution():
    SignalType1, IsPeriodic1, N11, signalm1 = read_file(st.session_state.timedomain)
    SignalType2, IsPeriodic2, N12, signalm2 = read_file(st.session_state.convolvetimedomain)
    signal = []
    min_index = int(signalm1[0, 0] + signalm2[0, 0])
    max_index = int(signalm1[-1, 0] + signalm2[-1, 0])
    for i in range(min_index, max_index + 1):
        convolutionsum = 0
        for sample1 in signalm1:
            for sample2 in signalm2:
                index = sample1[0] + sample2[0]
                if index == i:
                    convolutionsum += (sample1[1] * sample2[1])
        signal.append([i, convolutionsum])
    Signal = np.array(signal)
    plot_chart(Signal)
    if st.session_state.timedomain.name == "Input_conv_Sig1.txt":
        if st.session_state.convolvetimedomain.name == "Input_conv_Sig2.txt":
            ConvTest(Signal[:, 0], Signal[:, 1])
