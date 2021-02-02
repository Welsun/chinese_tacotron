from text import text_to_sequence
import hparams
from train import prepare_dataloaders
hp=hparams.create_hparams()
import glob
import os
import torch
import librosa
import soundfile as sf
from pinyin.parse_text_to_pyin import get_pyin
from data_utils import TextMelLoader
import matplotlib.pyplot as plt
mel_loader=TextMelLoader(hp.training_files,hp)
print(mel_loader[10])
# for wav in glob.glob("audio_segments/wavs/*.wav"):
#     wav,sr=librosa.load(wav,sr=22050)
#     print(len(wav))
#     sf.write("demo/test.wav",wav,samplerate=22050)
#     # mel_spec = mel_loader.get_mel(wav)
#     # plt.imsave("demo/ground_truth.jpg", mel_spec)
#     break

# print(len(glob.glob(hp.wav_dir+"/*.wav")))
# max_wav_value=0.0
# for i,name in enumerate(glob.glob(hp.wav_dir+"/*.wav")):
#     wav,_=librosa.load(name)
#     print(i)
#     print(wav.max())
#     max_wav_value=max(max_wav_value,wav.max())
#
# print(max_wav_value)

# train_loader,_,_=prepare_dataloaders(hp)
# print(train_loader)
# for batch in train_loader:
#     # print(train_loader.batch_size)
#     # print(len(batch))
#     text_padded, input_lengths, mel_padded, gate_padded, \
#     output_lengths = batch
#     print(text_padded.shape)
#     print(input_lengths.shape)
#     print(input_lengths)
#     break
#     # max_len=torch.max(input_lengths).item()
#     # ids = torch.arange(0, max_len, out=torch.LongTensor(max_len))
#     # print(ids)
#     # mask = (ids < input_lengths.unsqueeze(1)).bool()
#     # print(mask)
#     # print(mel_padded.shape)
#     # print(gate_padded.shape)
#     # print(gate_padded[:,1])
#     # print(output_lengths)
#     break

