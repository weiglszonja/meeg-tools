settings = dict(bandpass_filter=dict(low_freq=0.5, high_freq=45),
                epochs=dict(start_time=0., end_time=1.0, duration=1),
                ica=dict(n_components=32, method='infomax', decim=None),
                autoreject=dict(threshold=0.15))

tfr = dict(morlet=dict(fmin=4,
                       fmax=45,
                       step=30,
                       decim=1),
           baseline=dict(range=(-0.2, -0.1),
                         mode='mean'))
