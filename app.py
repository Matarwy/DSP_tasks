from methods import *
from comparesignals import SignalSamplesAreEqual
from os import path
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


with st.sidebar:
    select = st.radio("mode", ('Read Signal', 'Generate Signal', 'Arithmetic Operations'), index=0)
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
