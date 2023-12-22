import os.path

from methods import *
from comparesignals import SignalSamplesAreEqual
from QuanTest1 import *
from QuanTest2 import *
from Shift_Fold_Signal import *
from DerivativeSignal import DerivativeSignal
from CompareSignalFir import Compare_Signals as TestFir
# title
st.title("DSP Task 1")
if 'Addition_uploaded_files' not in st.session_state:
    st.session_state.Addition_uploaded_files = []
if 'Subtraction_Signal1' not in st.session_state:
    st.session_state.Subtraction_Signal1 = None
if 'Subtraction_Signal2' not in st.session_state:
    st.session_state.Subtraction_Signal2 = None
if 'Multiplication_Signal' not in st.session_state:
    st.session_state.Multiplication_Signal = None
if 'Multiplier' not in st.session_state:
    st.session_state.Multiplier = 1
if 'Squaring_Signal' not in st.session_state:
    st.session_state.Squaring_Signal = None
if 'Shifting_Signal' not in st.session_state:
    st.session_state.Shifting_Signal = None
if 'Shifter' not in st.session_state:
    st.session_state.Shifter = 0
if 'Normalization_Signal' not in st.session_state:
    st.session_state.Normalization_Signal = None
if 'Normalizer_min' not in st.session_state:
    st.session_state.Normalizer_min = -1
if 'Normalizer_max' not in st.session_state:
    st.session_state.Normalizer_max = 1
if 'Accumulation_Signal' not in st.session_state:
    st.session_state.Accumulation_Signal = None
if 'Quantization_file' not in st.session_state:
    st.session_state.Quantization_file = None
if 'num_of_levels' not in st.session_state:
    st.session_state.num_of_levels = 0
if 'dft_file' not in st.session_state:
    st.session_state.dft_file = None
if 'dft_fs' not in st.session_state:
    st.session_state.dft_fs = 0
if 'timedomain_signal' not in st.session_state:
    st.session_state.timedomain_signal = []
if 'freqdomain_signal' not in st.session_state:
    st.session_state.freqdomain_signal = []
if 'dc_file' not in st.session_state:
    st.session_state.dc_file = None
if 'timedomain' not in st.session_state:
    st.session_state.timedomain = None
if 'convolvetimedomain' not in st.session_state:
    st.session_state.convolvetimedomain = None
if 'dct_m' not in st.session_state:
    st.session_state.dct_m = 0
if 'corrlation' not in st.session_state:
    st.session_state.corrlation = []
if 'fastx' not in st.session_state:
    st.session_state.fastx = None
if 'fasth' not in st.session_state:
    st.session_state.fasth = None


with st.sidebar:
    select = st.radio(
        "mode",
        (
            'Read Signal',
            'Generate Signal',
            'Arithmetic Operations',
            'Quantization',
            'Frequency Domain',
            'Correlation',
            'Time Domain',
            'Practical'
        ),
        index=0)
if select == "Read Signal":
    with st.sidebar:
        uploaded_files = st.file_uploader("Choose your file", type="txt", key="load", accept_multiple_files=True)
    if len(uploaded_files) == 0:
        st.write("No Files Uploaded, Please Upload Signal File.")
    for uploaded_file in uploaded_files:
        SignalType, IsPeriodic, N1, Signal = read_file(uploaded_file)
        if SignalType == 0 and IsPeriodic == 0:
            plot_chart(Signal)
elif select == "Generate Signal":
    with st.sidebar:
        generate_type = st.radio("Type", ('sin', 'cos'), index=0)
        col1, col2 = st.columns(2)
        if generate_type == "sin":
            with col1:
                amplitude = st.number_input("Amplitude", min_value=1.0, value=3.0)
                phase_shift = st.number_input("phase shift theta", format="%f", value=1.96349540849362)
            with col2:
                analog_frequency = st.number_input("analog frequency", min_value=1.0, value=360.0)
                sampling_frequency = st.number_input("sampling frequency", min_value=analog_frequency/2, value=720.0)
        if generate_type == "cos":
            with col1:
                amplitude = st.number_input("Amplitude", min_value=1.0, value=3.0)
                phase_shift = st.number_input("phase shift theta", format="%f", value=2.35619449019235)
            with col2:
                analog_frequency = st.number_input("analog frequency", min_value=1.0, value=200.0)
                sampling_frequency = st.number_input("sampling frequency", min_value=analog_frequency / 2, value=500.0)
        generat_btn = st.button("Generate", type="primary")
    if generat_btn:
        Signal, N = generate_signal(generate_type, amplitude, phase_shift, analog_frequency, sampling_frequency)
        plot_chart(Signal)
        st.write(f"Number of samples: {N}")
        st.write(Signal)
elif select == "Arithmetic Operations":
    with st.sidebar:
        operation = st.radio(
            "Operation",
            ('Addition', 'Subtraction', 'Multiplication', 'Squaring', 'Shifting', 'Normalization', 'Accumulation'),
            index=0,
            horizontal=True,
            on_change=when_changed
        )

        if operation == "Addition":
            st.session_state.Addition_uploaded_files = st.file_uploader("Choose your file", type="txt", key="add", accept_multiple_files=True)
        if operation == "Subtraction":
            st.session_state.Subtraction_Signal1 = st.file_uploader("Signal 1", type="txt", key="sub1")
            st.session_state.Subtraction_Signal2 = st.file_uploader("Signal 2", type="txt", key="sub2")
        if operation == "Multiplication":
            st.session_state.Multiplication_Signal = st.file_uploader("Signal", type="txt", key="multiplication")
            st.session_state.Multiplier = st.number_input("Multiplier", format="%f", value=1.0)
        if operation == "Squaring":
            st.session_state.Squaring_Signal = st.file_uploader("Signal", type="txt", key="squaring")
        if operation == "Shifting":
            st.session_state.Shifting_Signal = st.file_uploader("Signal", type="txt", key="shifting")
            st.session_state.Shifter = st.number_input("Phase Shift", value=0)
        if operation == "Normalization":
            st.session_state.Normalization_Signal = st.file_uploader("Signal", type="txt", key="normalization")
            st.session_state.Normalizer_max = st.number_input("Max", value=st.session_state.Normalizer_max, min_value=st.session_state.Normalizer_min)
            st.session_state.Normalizer_min = st.number_input("Min", value=st.session_state.Normalizer_min, max_value=st.session_state.Normalizer_max)
        if operation == "Accumulation":
            st.session_state.Accumulation_Signal = st.file_uploader("Signal", type="txt", key="accumulation")
    if len(st.session_state.Addition_uploaded_files) > 1:
        Signal = Addition(st.session_state.Addition_uploaded_files)
        plot_chart(Signal)
        if st.session_state.Addition_uploaded_files[0].name == "Signal1.txt":
            if st.session_state.Addition_uploaded_files[1].name == "Signal2.txt":
                print("Signal1.txt + Signal2.txt")
                SignalSamplesAreEqual(
                    path.relpath("signals/output signals/Signal1+signal2.txt"),
                    Signal[:, 0],
                    Signal[:, 1]
                )
            if st.session_state.Addition_uploaded_files[1].name == "signal3.txt":
                print("Signal1.txt + Signal3.txt")
                SignalSamplesAreEqual(
                    path.relpath("signals/output signals/signal1+signal3.txt"),
                    Signal[:, 0],
                    Signal[:, 1]
                )
    if st.session_state.Subtraction_Signal1 is not None and st.session_state.Subtraction_Signal2 is not None:
        Signal = Subtraction(st.session_state.Subtraction_Signal1, st.session_state.Subtraction_Signal2)
        plot_chart(Signal)
        if st.session_state.Subtraction_Signal1.name == "Signal1.txt":
            if st.session_state.Subtraction_Signal2.name == "Signal2.txt":
                print("Signal1.txt - Signal2.txt")
                SignalSamplesAreEqual(
                    path.relpath("signals/output signals/signal1-signal2.txt"),
                    Signal[:, 0],
                    Signal[:, 1]
                )
            if st.session_state.Subtraction_Signal2.name == "signal3.txt":
                print("Signal1.txt - Signal3.txt")
                SignalSamplesAreEqual(
                    path.relpath("signals/output signals/signal1-signal3.txt"),
                    Signal[:, 0],
                    Signal[:, 1]
                )
    if st.session_state.Multiplication_Signal is not None:
        Signal = Multiplication(st.session_state.Multiplication_Signal, st.session_state.Multiplier)
        plot_chart(Signal)
        if st.session_state.Multiplication_Signal.name == "Signal1.txt":
            if st.session_state.Multiplier == 5:
                print("Signal1.txt Multiply by 5")
                SignalSamplesAreEqual(
                    path.relpath("signals/output signals/MultiplySignalByConstant-Signal1 - by 5.txt"),
                    Signal[:, 0],
                    Signal[:, 1]
                )
        if st.session_state.Multiplication_Signal.name == "Signal2.txt":
            if st.session_state.Multiplier == 10:
                print("Signal2.txt Multiply by 10")
                SignalSamplesAreEqual(
                    path.relpath("signals/output signals/MultiplySignalByConstant-signal2 - by 10.txt"),
                    Signal[:, 0],
                    Signal[:, 1]
                )
    if st.session_state.Squaring_Signal is not None:
        Signal = Squaring(st.session_state.Squaring_Signal)
        plot_chart(Signal)
        if st.session_state.Squaring_Signal.name == "Signal1.txt":
            print("Signal1.txt Squaring")
            SignalSamplesAreEqual(
                path.relpath("signals/output signals/Output squaring signal 1.txt"),
                Signal[:, 0],
                Signal[:, 1]
            )
    if st.session_state.Shifting_Signal is not None:
        Signal = Shifting(st.session_state.Shifting_Signal, st.session_state.Shifter)
        plot_chart(Signal)
        if st.session_state.Shifting_Signal.name == "Input Shifting.txt":
            if st.session_state.Shifter == 500:
                print("Shifting.txt Shifting by add 500")
                SignalSamplesAreEqual(
                    path.relpath("signals/output signals/output shifting by add 500.txt"),
                    Signal[:, 0],
                    Signal[:, 1]
                )
            if st.session_state.Shifter == -500:
                print("Shifting.txt Shifting by minus 500")
                SignalSamplesAreEqual(
                    path.relpath("signals/output signals/output shifting by minus 500.txt"),
                    Signal[:, 0],
                    Signal[:, 1]
                )
    if st.session_state.Normalization_Signal is not None:
        Signal = Normalization(st.session_state.Normalization_Signal, st.session_state.Normalizer_max, st.session_state.Normalizer_min)
        plot_chart(Signal)
        if st.session_state.Normalization_Signal.name == "Signal1.txt":
            if st.session_state.Normalizer_max == 1 and st.session_state.Normalizer_min == -1:
                print("Normalize Signal1.txt max=1 min=-1")
                SignalSamplesAreEqual(
                    path.relpath("signals/output signals/normalize of signal 1 -- output.txt"),
                    Signal[:, 0],
                    Signal[:, 1]
                )
        if st.session_state.Normalization_Signal.name == "Signal2.txt":
            if st.session_state.Normalizer_max == 1 and st.session_state.Normalizer_min == 0:
                print("Normalize Signal1.txt max=1 min=0")
                SignalSamplesAreEqual(
                    path.relpath("signals/output signals/normlize signal 2 -- output.txt"),
                    Signal[:, 0],
                    Signal[:, 1]
                )
    if st.session_state.Accumulation_Signal is not None:
        Signal = Accumulation(st.session_state.Accumulation_Signal)
        plot_chart(Signal)
        if st.session_state.Accumulation_Signal.name == "Signal1.txt":
            print("accumulation for signal1.txt")
            SignalSamplesAreEqual(
                path.relpath("signals/output signals/output accumulation for signal1.txt"),
                Signal[:, 0],
                Signal[:, 1]
            )
elif select == "Quantization":
    with st.sidebar:
        st.session_state.Quantization_file = st.file_uploader("Signal", type="txt", key="quantization")
        lev_type = st.radio("choose input type", ("Levels", "Bits"), index=0)
    if lev_type == "Levels":
        with st.sidebar:
            st.session_state.num_of_levels = int(st.number_input("levels", min_value=1, key="levels"))

    elif lev_type == "Bits":
        with st.sidebar:
            st.session_state.num_of_levels = 2 ** st.number_input("levels", min_value=1, key="bits")
    if st.session_state.Quantization_file is not None:
        interval, encoded, quntized, errors = Quantization(
            st.session_state.Quantization_file,
            st.session_state.num_of_levels
        )
        # output_file_1 = path.relpath('signals/task3/Test 1/Quan1_Out.txt')
        output_file_2 = path.relpath('signals/task3/Test 2/Quan2_Out.txt')
        # QuantizationTest1(output_file_1, encoded, quntized[:, 1])
        QuantizationTest2(output_file_2, interval, encoded, quntized[:, 1], errors[:, 1])
        chart = make_subplots(rows=1, cols=1)
        chart.add_trace(go.Scatter(y=quntized[:, 1], x=quntized[:, 0], mode="lines", name="Quntized"), row=1, col=1)

        chart.add_trace(
            go.Scatter(
                y=quntized[:, 1],
                x=quntized[:, 0],
                error_y=dict(
                    type='data',
                    array=errors[:, 1],
                    visible=True,
                    symmetric=False
                ),
                mode="markers",
                name="ERROR"),
            row=1, col=1)
        chart.update_xaxes(title_text='Index')
        chart.update_yaxes(title_text='Level')
        chart.update_layout(autosize=True)
        st.plotly_chart(chart, use_container_width=True)
elif select == "Frequency Domain":
    with st.sidebar:
        freqselect = st.radio("choose operation", ("dft", "dct", "fast convolution", "fast correlation"), index=0)
        if freqselect == "dft" or freqselect == "dct":
            dc_checkbox = st.checkbox("Remove DC Component")
            st.session_state.dft_file = st.file_uploader("Signal", type="txt", key="dft")
            st.session_state.dft_fs = st.number_input("FS", value=1, min_value=1)
        if freqselect == "fast convolution" or freqselect == "fast correlation":
            st.session_state.fastx = st.file_uploader("X", type="txt", key="fastdftx")
            st.session_state.fasth = st.file_uploader("H", type="txt", key="fastdfth")
    if st.session_state.dft_file is not None:
        SignalType, IsPeriodic, N1, signalm = read_file(st.session_state.dft_file)
        remove_dc = False
        if dc_checkbox:
            remove_dc = True
        if freqselect == "dft":
            dft(SignalType=SignalType, N1=N1, signal=signalm, fs=st.session_state.dft_fs, dc=remove_dc)
        elif freqselect == "dct":
            dft(SignalType=SignalType, N1=N1, signal=signalm, fs=st.session_state.dft_fs, operation="dct", dc=remove_dc)
            with st.sidebar:
                st.session_state.dct_m = st.number_input("M", value=N1, min_value=0, max_value=N1, key="mcoff")
        freqamplitudevalue = ""
        freqphasevalue = ""
        for freq in st.session_state.freqdomain_signal:
            freqamplitudevalue += f"{freq[1]}\n"
            freqphasevalue += f"{freq[2]}\n"

        with st.sidebar:
            col1, col2 = st.columns(2)
            with col1:
                write_file = st.button("Write to TXT", type="primary")
            with col2:
                update_signal = st.button("Update", type="primary")
            freqamplitude = st.text_area('Update the Amplitude', value=freqamplitudevalue)
            freqphase = st.text_area('Update the Phase shift', value=freqphasevalue)

        if write_file:
            # Open a text file for writing
            with open("result.txt", "w") as file:
                if freqselect == "dft":
                    lines = f"1\n0\n{len(st.session_state.freqdomain_signal)}\n"
                    for freq in st.session_state.freqdomain_signal:
                        lines += f"{freq[0]} {freq[1]} {freq[2]}\n"
                elif freqselect == "dct":
                    lines = f"1\n0\n{st.session_state.dct_m}\n"
                    for freq in st.session_state.freqdomain_signal[:st.session_state.dct_m]:
                        lines += f"{freq[0]} {freq[1]} {freq[2]}\n"

                # Write the square root value to the file
                file.write(lines)

        if update_signal:
            rowsamp = freqamplitude.strip().split("\n")
            rowsphase = freqphase.strip().split("\n")
            updated_array_amp = [list(map(float, row.strip().split())) for row in rowsamp]
            updated_array_phase = [list(map(float, row.strip().split())) for row in rowsphase]
            updated_array_amp = np.array(updated_array_amp)
            updated_array_phase = np.array(updated_array_phase)
            if len(updated_array_phase) == len(updated_array_amp):
                new_freqdomain_signal = []
                for i in range(0, len(updated_array_amp)):
                    newFreq = i * st.session_state.freqdomain_signal[1,0]
                    new_freqdomain_signal.append([
                        newFreq,
                        updated_array_amp[i, 0],
                        updated_array_phase[i, 0]
                    ])
                new_freqdomain_signal = np.array(new_freqdomain_signal)
                st.session_state.freqdomain_signal = new_freqdomain_signal
                dft(
                    SignalType=1,
                    N1=len(st.session_state.freqdomain_signal),
                    signal=st.session_state.freqdomain_signal,
                    fs=st.session_state.dft_fs
                )
            else:
                print("Error, while updating signal")
        st.header("Time Domain")
        plot_chart(st.session_state.timedomain_signal)
        st.header("Frequency Domain")
        plot_bar(st.session_state.freqdomain_signal)
    if st.session_state.fastx is not None:
        if st.session_state.fasth is not None:
            if freqselect == "fast convolution":
                fastconvv()
            elif freqselect == "fast correlation":
                SignalType1, IsPeriodic1, N11, signalm1 = read_file(st.session_state.fastx)
                SignalType2, IsPeriodic2, N12, signalm2 = read_file(st.session_state.fasth)
                resault_signal = fastcorr(signalm1, signalm2)
                plot_chart(resault_signal)
elif select == "Correlation":
    with st.sidebar:
        corr_mode = st.radio("Select", ("Cross-Croolation", "Time Analysis", "Template matching"))
        if corr_mode == "Cross-Croolation" or corr_mode == "Time Analysis":
            st.session_state.corrlation = st.file_uploader("Signal", type="txt", key="corr1", accept_multiple_files=True)

    if len(st.session_state.corrlation) > 1:
        SignalType1, IsPeriodic1, N11, signalm1 = read_file(st.session_state.corrlation[0])
        SignalType2, IsPeriodic2, N12, signalm2 = read_file(st.session_state.corrlation[1])
        R12 = cross_correlation(signalm1, signalm2)
        if st.session_state.corrlation[0].name == 'Corr_input signal1.txt':
            if st.session_state.corrlation[1].name == 'Corr_input signal2.txt':
                cs_comp(path.relpath("signals/Task7/Point1 Correlation/CorrOutput.txt"), R12[:, 0], R12[:, 1])
        plot_chart(R12)
        if corr_mode == "Time Analysis":
            with st.sidebar:
                Fs = int(st.number_input("FS", value=100))
            delay = corr_time_analysis(R12, Fs)
            st.write(f"Time Delay: %s" % delay)
    elif corr_mode == "Template matching":
        with st.sidebar:
            class1 = st.text_input(
                'Class 1 path', "C:\\Users\Matarwy\Documents\FCIS\DSP\Labs\Lab7\SC and Csys\Task7\point3 Files\Class 1")
            class2 = st.text_input(
                'Class 2 path', "C:\\Users\Matarwy\Documents\FCIS\DSP\Labs\Lab7\SC and Csys\Task7\point3 Files\Class 2")
            template = st.text_input(
                'Test path', 'C:\\Users\Matarwy\Documents\FCIS\DSP\Labs\Lab7\SC and Csys\Task7\point3 Files\Test Signals')
            button_click = st.button('Match', type="primary")
        if button_click:
            template_matching(template, class1, class2)
elif select == "Time Domain":
    with st.sidebar:
        st.session_state.timedomain = st.file_uploader("Signal", type="txt", key="timedo")
        smoothing = st.toggle("Smoothing")
        sharpening = st.toggle("Sharpening")
        delayingoradvancing = st.toggle("Delaying or Advancing")
        folding = st.toggle("Folding")
        convolve = st.toggle("Convolution")
        filtering = st.toggle("Filtering")
        resample = st.toggle("Resampling")
    if smoothing:
        with st.sidebar:
            st.header("Smoothing")
            smothingws = st.number_input("Smoothing Window Size", min_value=0, value=3)
        if st.session_state.timedomain is not None:
            timedomain_smoothing(int(smothingws))
    if sharpening:
        DerivativeSignal()
    if delayingoradvancing and not folding:
        with st.sidebar:
            st.header("Delaying or Advancing")
            daak = st.number_input("K", value=0)
        if st.session_state.timedomain is not None:
            daasignal = timedomain_delaying_advance(daak)
            plot_chart(daasignal)
    if folding and not delayingoradvancing:
        if st.session_state.timedomain is not None:
            SignalType, IsPeriodic, N1, signalm = read_file(st.session_state.timedomain)
            daasignal = timedomain_folding(signalm)
            plot_chart(daasignal)
            if st.session_state.timedomain.name == "input_fold.txt":
                Shift_Fold_Signal(
                    path.relpath("signals/lab6/Shifting and Folding/Output_fold.txt"),
                    daasignal[:, 0],
                    daasignal[:, 1]
                )
    if delayingoradvancing and folding:
        with st.sidebar:
            st.header("Delaying or Advancing")
            daak = st.number_input("K", value=0)
        if st.session_state.timedomain is not None:
            daasignal = timedomain_delaying_advance(daak)
            daasignal = timedomain_folding(daasignal)
            plot_chart(daasignal)
            if st.session_state.timedomain.name == "input_fold.txt" and daak == 500:
                Shift_Fold_Signal(
                    path.relpath("signals/lab6/Shifting and Folding/Output_ShifFoldedby500.txt"),
                    daasignal[:, 0],
                    daasignal[:, 1]
                )
            if st.session_state.timedomain.name == "input_fold.txt" and daak == -500:
                Shift_Fold_Signal(
                    path.relpath("signals/lab6/Shifting and Folding/Output_ShiftFoldedby-500.txt"),
                    daasignal[:, 0],
                    daasignal[:, 1]
                )
    if convolve:
        with st.sidebar:
            st.header("Convolution")
            st.session_state.convolvetimedomain = st.file_uploader("Signal", type="txt", key="convolve")
        if st.session_state.convolvetimedomain is not None:
            if st.session_state.timedomain is not None:
                SignalType1, IsPeriodic1, N11, signalm1 = read_file(st.session_state.timedomain)
                SignalType2, IsPeriodic2, N12, signalm2 = read_file(st.session_state.convolvetimedomain)
                ResSignal = timedomain_convolution(signalm1, signalm2)
                plot_chart(ResSignal)
    if filtering:
        with st.sidebar:
            filtertype = st.selectbox(
                'Filter Type?',
                ('Low Pass', 'High Pass', 'Band Pass', 'Band Stop'))
            sampling_frequency = st.number_input('Sampling Frequency')
            stop_attenuation = st.number_input('δs')
            transition_band = st.number_input('Transition Band')
        hn = []
        if filtertype == 'Low Pass' or filtertype == 'High Pass':
            with st.sidebar:
                cut_off_frequancy = st.number_input('Cut off Frequency')
            if cut_off_frequancy != 0:
                hn = fir_hn_w(
                    filtertype,
                    sampling_frequency,
                    transition_band,
                    stop_attenuation,
                    cutofreq=cut_off_frequancy
                )
        if filtertype == 'Band Pass' or filtertype == 'Band Stop':
            with st.sidebar:
                f1 = st.number_input("F1")
                f2 = st.number_input("F2")
            if f1 != 0 and f2 != 0:
                hn = fir_hn_w(
                    filtertype,
                    sampling_frequency,
                    transition_band,
                    stop_attenuation,
                    cutofreq1=f1,
                    cutofreq2=f2
                )
        if len(hn) != 0:
            st.header(f"Filter {filtertype}")
            plot_chart(hn)
            np.savetxt('coefficients.txt', hn)

        if st.session_state.timedomain is not None:
            if len(hn) != 0:
                st.header(f"Filtered signal")
                SignalType1, IsPeriodic1, N11, signalm1 = read_file(st.session_state.timedomain)
                ResSignal = timedomain_convolution(signalm1, hn)
                plot_chart(ResSignal)
        with st.sidebar:
            runtests = st.button('Run tests', type="primary")
        if runtests:
            hn = fir_hn_w(filter_type="High Pass", fs=8000, transition_band=500, stop_attenuation=70, cutofreq=1500)
            print(hn)
            TestFir(
                os.path.relpath("signals/practical/FIR test cases/Testcase 3/HPFCoefficients.txt"),
                hn[:, 0], hn[:, 1],
            )

            hn = fir_hn_w(filter_type="Low Pass", fs=8000, transition_band=500, stop_attenuation=50, cutofreq=1500)
            # print(hn)
            TestFir(
                os.path.relpath("signals/practical/FIR test cases/Testcase 1/LPFCoefficients.txt"),
                hn[:, 0], hn[:, 1],
            )
            hn = fir_hn_w(
                filter_type="Band Pass", fs=1000, transition_band=50, stop_attenuation=60, cutofreq1=150, cutofreq2=250,)
            TestFir(
                os.path.relpath("signals/practical/FIR test cases/Testcase 5/BPFCoefficients.txt"),
                hn[:, 0], hn[:, 1],
            )
            # print(hn)
            hn = fir_hn_w(filter_type="Band Stop", fs=1000, transition_band=50, stop_attenuation=60, cutofreq1=150, cutofreq2=250)
            TestFir(
                os.path.relpath("signals/practical/FIR test cases/Testcase 7/BSFCoefficients.txt"),
                hn[:, 0], hn[:, 1],
            )
            print(hn)
    if resample:
        with st.sidebar:
            sampling_frequency = st.number_input('Sampling Frequency', value=8000)
            stop_attenuation = st.number_input('δs', value=50)
            transition_band = st.number_input('Transition Band', value=500)
            cut_off_frequancy = st.number_input('Cut off Frequency', value=1500)
            L = int(st.number_input('L'))
            M = int(st.number_input('M'))
            # runTests = st.number_input('Run Tests', type='primary')
        if st.session_state.timedomain is not None:
            SignalType1, IsPeriodic1, N11, signalm1 = read_file(st.session_state.timedomain)
            hn = fir_hn_w(
                'Low Pass',
                sampling_frequency,
                transition_band,
                stop_attenuation,
                cutofreq=cut_off_frequancy
            )
            if L == 0 and M == 0:
                st.write("Error L = 0 and M = 0")
            elif L != 0 and M == 0:
                upsample = []
                index = signalm1[0, 0]
                for sampl in signalm1:
                    upsample.append([index, sampl[1]])
                    index += 1
                    for i in range(0, L - 1):
                        upsample.append([index, 0])
                        index += 1
                UpSample = np.array(upsample)
                ResSignal = timedomain_convolution(UpSample, hn)
                plot_chart(ResSignal)
            elif L == 0 and M != 0:
                downsample = []
                ResSignal = timedomain_convolution(signalm1, hn)
                for i in range(0, len(ResSignal)):
                    downsample.append([ResSignal[i, 0], ResSignal[i, 1]])
                    i += M
                DownSample = np.array(downsample)
                st.header("DownSample")
                plot_chart(DownSample)
            elif L != 0 and M != 0:
                upsample = []
                downsample = []
                index = signalm1[0, 0]
                for sampl in signalm1:
                    upsample.append([index, sampl[1]])
                    index += 1
                    for i in range(0, L - 1):
                        upsample.append([index, 0])
                        index += 1
                UpSample = np.array(upsample)
                ResSignal = timedomain_convolution(UpSample, hn)
                i = 0
                for i in range(0, int(len(ResSignal) / M)):
                    downsample.append([ResSignal[i, 0], ResSignal[i*M, 1]])

                DownSample = np.array(downsample)
                print(DownSample)
                st.header("DownSample")
                plot_chart(DownSample)
                if st.session_state.timedomain.name == "ecg400.txt":
                    TestFir(
                        os.path.relpath("signals/practical/Sampling test cases/Testcase 3/Sampling_Up_Down.txt"),
                        DownSample[:, 0],
                        DownSample[:, 1]
                    )

elif select == "Practical":
    with st.sidebar:
        a_path = st.text_input(
            "A Path","C:\\Users\Matarwy\Documents\FCIS\DSP\Labs\Practical Task\Practical task 2\A"
        )
        b_path = st.text_input(
            "B Path", "C:\\Users\Matarwy\Documents\FCIS\DSP\Labs\Practical Task\Practical task 2\B"
        )
        test_path = st.text_input(
            "Test Folder", "C:\\Users\Matarwy\Documents\FCIS\DSP\Labs\Practical Task\Practical task 2\Test Folder"
        )
        Fs = st.number_input("Fs", value=8000)
        minF = st.number_input("minF", value=1500)
        maxF = st.number_input("maxF", value= 1800)
        L = st.number_input("L")
        M = st.number_input("M")
        button_click = st.button('Start', type="primary")
    if button_click:
        practice(a_path, b_path, test_path, Fs, minF, maxF, L, M)


