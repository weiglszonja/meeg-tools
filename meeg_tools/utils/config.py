settings = dict(
    log=dict(author='Szonja Weigl'),
    bandpass_filter=dict(low_freq=0.5, high_freq=45),
    epochs=dict(start_time=-0.700, end_time=0.750, duration=1),
    ica=dict(n_components=32, method="picard", decim=None),
    autoreject=dict(threshold=0.15),
)
