from mfcc_io import mfcc
import librosa
# audio_file ='../data_feature_script/test123.wav'
audio_file = '/mnt/d/project/1.6svoice/HeyMemo/AUS_Sydney_Female_28_HeyMemo_-24.3dB_3.1wps.wav'
audio_data = librosa.load(audio_file, sr=16000)[0]
mfcc_data = mfcc(y=audio_data, sr=16000, n_mfcc=16, n_mels=40, S=None, norm=None,
                           win_length=512, window='hamming', hop_length=256, n_fft=512,
                           fmin=20, fmax=4050, center=False, power=1, htk=True, dct_type=2, lifter=0,
                           scale_exp=15)
# print(mfcc_data.shape)
# print(mfcc_data)