import torch
import torchaudio
import matplotlib.pyplot as plt



# 我们用一个原始音频信号或波形的例子来说明如何使用torchaudio打开音频文件，
# 以及如何预处理和转换这样的波形。由于torchaudio是构建在PyTorch之上的，
# 因此这些技术可以作为更高级的音频应用程序(如语音识别)的构建块，
# 同时利用gpu

# Let's normalize to the full interval [-1,1]
def normalize(tensor):
    # Subtract the mean, and scale to the interval [-1,1]
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean / tensor_minusmean.abs().max()

    pass
def tryAudio_1():

    # Opening a dataset
    filename = "/home/ZhangXueLiang/LiMiao/dataset/Tutorial/下一个天亮.mp3"
    waveform,sample_rate = torchaudio.load(filename)
    # print("Shape of waveform: {}".format(waveform.size()))
    # print("Sample rate of waveform: {}".format(sample_rate))
    # plt.figure(1)
    # plt.plot(waveform.t().numpy())
    # plt.savefig('waveform.jpg')
    #
    #
    # # Transformaations
    # # Spectrogram: Create a spectrogram from a waveform.
    # specgram =torchaudio.transforms.Spectrogram()(waveform)
    # print("Shape of spectrogram: {}".format(specgram.size()))
    # plt.figure(2)
    # plt.imshow(specgram.log2()[0,:,:].numpy(),cmap='Blues_r')
    # # plt.plot(specgram.log2()[0,:,:].numpy())
    # plt.savefig('transformswave.jpg')
    #
    # # Or we can look at the Mel Spectrogram on a log scale
    # specgram = torchaudio.transforms.MelSpectrogram()(waveform)
    # print("Shape of spectrogram: {}".format(specgram.size()))
    # plt.figure(3)
    # p = plt.imshow(specgram.log2()[0,:,:].detach().numpy(),cmap='gray')
    # # p = plt.plot(specgram.log2()[0,:,:].detach().numpy())
    # plt.savefig('transformsMelwave,jpg')
    #
    #
    # # We can resample the waveform, one channel at a time.
    # new_sample_rate = sample_rate/10
    # # Since Resample applies to a single channel, we reshample first channel here
    # channel = 0
    # transformed = torchaudio.transforms.Resample(sample_rate,new_sample_rate)(waveform[channel,:].view(1,-1))
    # print("Shape of transformed waveform: {}".format(transformed.size()))
    # plt.figure(4)
    # plt.plot(transformed[0,:].numpy())
    # plt.savefig('new_trnasform.jpg')
    #
    # # Let's check if the tensor is in the interval [-1,1]
    # print("Min of waveform: {}\nMax of waveform: {}\nMean of waveform:{}".format(waveform.min(),waveform.max(),waveform.mean()))
    #
    # # Let’s apply encode the waveform.
    transformed = torchaudio.transforms.MuLawEncoding()(waveform)
    # print("Shpae of transformed waveform: {}".format(transformed.size()))
    # plt.figure(4)
    # plt.plot(transformed[0,:].numpy())
    # plt.savefig('encodetransformwave.jpg')

    # And now decode.
    reconstructed = torchaudio.transforms.MuLawDecoding()(transformed)
    print("Shape of recovered waveform: {}".format(reconstructed.size()))

    plt.figure(5)
    plt.plot(reconstructed[0,:].numpy())
    plt.savefig('decodetransformwave.jpg')

    # We can finally compare the original waveform with its reconstructed version.
    err = ((waveform-reconstructed).abs() / waveform.abs()).median()
    print("Median relative difference between original and MuLaw reconstructed signals: {:.2%}".format(err))

    n_fft = 400.0
    frame_length = n_fft / sample_rate*1000.0
    frame_shift = frame_length / 2.0

    params = {
        "channel": 0,
        "dither": 0.0,
        "window_type":"hanning",
        "frame_length":frame_length,
        "frame_shift":frame_shift,
        "remove_dc_offset":False,
        "round_to_power_of_two":False,
        "sample_frequency":sample_rate,
    }
    specgram = torchaudio.compliance.kaldi.spectrogram(waveform,**params)
    print("Shape of spectrogram: {}".format(specgram.size()))
    plt.figure(6)
    plt.imshow(specgram.t().numpy(),cmap='gray')
    plt.savefig('kaldiwave.jpg')

    # We also support computing the filterbank features from waveforms, matching Kaldi’s implementation.
    fbank = torchaudio.compliance.kaldi.fbank(waveform,**params)
    print("Shape of fbank: {}".format(fbank.size()))
    plt.figure(7)
    plt.imshow(fbank.t().numpy(),cmap='gray')
    plt.savefig('fbankwave.jpg')



    pass




def tryAudio_2():


    pass
def main():
    tryAudio_1()
    pass

if __name__ == '__main__':
    main()
