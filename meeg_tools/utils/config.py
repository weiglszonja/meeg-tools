settings = dict(bandpass_filter=dict(low_freq=0.5, high_freq=45),
                epochs=dict(start_time=0., end_time=1.0, duration=1),
                ica=dict(n_components=32, method='picard', decim=None),
                autoreject=dict(threshold=0.15))

analysis = dict(morlet=dict(fmin=4,
                            fmax=45,
                            step=30,
                            decim=1),
                bands=dict(theta=(4.0, 8.0),
                           alpha=(8.0, 13.0),
                           beta=(13.0, 30.0),
                           gamma=(30.0, 45.0)),
                baseline=dict(range=None,
                              mode='logratio'))
