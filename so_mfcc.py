from mfcc_io import mfcc
import librosa
# audio_file ='../data_feature_script/test123.wav'
audio_file = '/mnt/f/string_test/heymemo/SPK020_DEU_Offenburg_Male_25_HeyMemo_-17.8dB_2.1wps.wav_padded_snr10.wav'
audio_data = librosa.load(audio_file, sr=16000)[0]
mfcc_data = mfcc(y=audio_data, sr=16000, n_mfcc=16, n_mels=40, S=None, norm=None,
                           win_length=512, window='hamming', hop_length=256, n_fft=512,
                           fmin=20, fmax=4050, center=False, power=1, htk=True, dct_type=2, lifter=0,
                           scale_exp=15)
print(mfcc_data.shape)
print(mfcc_data)

'''
### 基本参数
- y=audio_data : 这是您要处理的音频时间序列，也就是原始的音频波形数据。
- sr=16000 : 这是音频的采样率（Sample Rate），单位是赫兹（Hz）。它表示每秒钟从连续信号中提取并组成离散信号的采样个数。16000Hz是语音处理中常用的采样率。
- n_mfcc=16 : 这是您希望返回的MFCC系数的数量。MFCC是一组特征，这个参数决定了特征向量的维度。

### 梅尔频谱图相关参数
这些参数用于计算MFCC过程中的中间步骤——梅尔频谱图。
- n_mels=40 : 指定在梅尔刻度上使用的滤波器组（Mel bands）的数量。这个值越大，频率分辨率越高。
- win_length=512 : 这是傅里叶变换（FFT）的窗口大小，单位是采样点数。它决定了频谱分析的时间分辨率和频率分辨率。
- window='hamming' : 指定在分帧时使用的窗函数。窗函数用于减少频谱泄漏。'hamming'是一种常见的选择。
- hop_length=256 : 这是相邻帧之间的步长，单位是采样点数。它决定了特征的时间分辨率。这个值越小，时间分辨率越高，但计算量也越大。
- n_fft=512 : 这是执行FFT的点数。通常， n_fft 大于等于 win_length 。如果 n_fft > win_length ，则会对窗口化的帧进行零填充。
- fmin=20 : 计算梅尔滤波器组时的最低频率，单位是Hz。
- fmax=4050 : 计算梅尔滤波器组时的最高频率，单位是Hz。
- power=1 : 在计算梅尔频谱图之前应用于频谱的指数。1表示使用幅度谱，2表示使用功率谱（默认值）。

### DCT 和 后处理参数
- dct_type=2 : 指定离散余弦变换（DCT）的类型。DCT用于将对数梅尔频谱转换为MFCC。类型2是标准的DCT。
- norm=None : 指定DCT是否使用正交归一化。 'ortho' 会使用正交基。在您的代码中，它被设置为 None 。
- lifter=0 : 倒谱提升（Cepstral liftering）系数。如果大于0，它会应用于MFCC以调整高阶系数的权重。 0 表示不进行倒谱提升。

### 其他参数
- S=None : 这是一个可选参数，可以直接传入一个预先计算好的对数功率梅尔频谱图，而不是从原始音频 y 开始计算。这里是 None ，所以会从 y 开始计算。
- center=False : 如果为 True ，信号会被填充，以使帧 t 的中心位于 y[t * hop_length] 。如果为 False ，则帧 t 从 y[t * hop_length] 开始。
- htk=True : 如果为 True ，则使用HTK（Hidden Markov Model Toolkit）风格的梅尔滤波器组公式，而不是默认的Slaney风格。这两种风格在梅尔频率的计算上略有不同。
- scale_exp=15 : 这个参数不是 librosa.feature.mfcc 的标准参数。很可能您使用的是一个自定义版本的 mfcc 函数，或者这个参数是通过 **kwargs 传递给其他内部函数的。如果这是您自己实现的 mfcc 函数，您需要查看其内部实现来确定 scale_exp 的具体作用。它可能用于对最终的MFCC系数进行缩放。
'''
