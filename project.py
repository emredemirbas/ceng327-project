import os
import matplotlib.pyplot as plt
from mne.datasets.sleep_physionet.age import fetch_data
import mne

# Sadece bir katılımcının verisini indiriyor
subjects = [0]  # İlk katılımcı
recordings = [1]  # Birinci kayıt
fnames = fetch_data(subjects=subjects, recording=recordings, on_missing='warn')

# Veriyi yükleme fonksiyonu
def load_sleep_physionet_raw(raw_fname, annot_fname, load_eeg_only=True):
    mapping = {'EOG horizontal': 'eog',
               'Resp oro-nasal': 'misc',
               'EMG submental': 'misc',
               'Temp rectal': 'misc',
               'Event marker': 'misc'}
    exclude = mapping.keys() if load_eeg_only else ()
    
    raw = mne.io.read_raw_edf(raw_fname, exclude=exclude)
    annots = mne.read_annotations(annot_fname)
    raw.set_annotations(annots, emit_warning=False)
    return raw

# Veriyi yükleme
raw_fname, annot_fname = fnames[0]
raw = load_sleep_physionet_raw(raw_fname, annot_fname)

# EEG kanallarını seçme
eeg_channels = mne.pick_types(raw.info, eeg=True)

raw.crop(tmax=600)  
data, times = raw[eeg_channels]

# EEG sinyalini görselleştirme
plt.figure(figsize=(15, 5))
plt.plot(times, data.T) 
plt.title("EEG Signal (First 10 Minutes)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (uV)")
plt.grid(True)
plt.show()
