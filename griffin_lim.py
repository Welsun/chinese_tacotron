from audio_processing import griffin_lim,dynamic_range_compression
from data_utils import TextMelLoader
from hparams import create_hparams
import os
import matplotlib

import matplotlib.pyplot as plt

import torch
import numpy as np
from audio_processing import griffin_lim
import librosa
import soundfile
import torch

from model import Tacotron2
from train import load_model
from text import text_to_sequence
from pinyin.parse_text_to_pyin import get_pyin
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

def mel2audio(mel):
    hp = create_hparams()
    mel_loader = TextMelLoader(hp.training_files, hp)
    ## denormlize
    mel_spec = mel_loader.stft.spectral_de_normalize(mel)

    inv_basis = torch.tensor(np.linalg.pinv(mel_loader.stft.mel_basis))
    mag = torch.matmul(inv_basis, mel_spec)

    signal = griffin_lim(mag.unsqueeze(0), mel_loader.stft.stft_fn, n_iters=60).squeeze(0).numpy()
    print(signal.shape)
    return signal

hp=create_hparams()
hp.distributed_run=False

with open("filelists/news_test.txt") as f:
    files=f.readlines()
wav,text=files[0].split("|")
print(wav)
print(text)
print(text)
file=os.path.join(hp.wav_dir,wav)
print(file)
mel_loader=TextMelLoader(hp.training_files,hp)
mel_spec=mel_loader.get_mel(file)

plt.imsave("demo/ground_truth.jpg",mel_spec,cmap="hot")
checkpoint='news_output_22k/checkpoint_15000'
model=load_model(hp)
model.load_state_dict(torch.load(checkpoint)['state_dict'])
_=model.cuda().eval().half()
#text="ka2 er2 pu3 pei2 wai4 sun1 wan2 hua2 ti1"
#text="bu4 xiao3 xin1 nong4 huai4 lao3 ba4 de5 an4 mo2 yi3 lao3 ba4 dui4 wo3 shuo1 ni3 ruo4 an1 hao3 bian4 shi4 qing2 tian1"
#text="zhong1 zhen4 tao1 yu2 nv3 er2 qian1 shou3 guang4 jie1"
#text="在狱中，张明宝悔恨交加，写了一份忏悔书。"
#text="我我我我我，你你你你，我不知道怎么说出口。"


text,_=get_pyin(text)
print(text)

sequence=np.array(text_to_sequence(text,['basic_cleaners']))[None,:]
sequence=torch.autograd.Variable(torch.from_numpy(sequence).cuda().long())
mel_output,mel_output_posnet,_,alignment=model.inference(sequence)

mel_output=mel_output.float().data.cpu()[0]
mel_output_posnet=mel_output_posnet.float().data.cpu()[0]
align=alignment.float().data.cpu()[0].T

plt.imsave("demo/pred.jpg",mel_output,cmap="hot")
plt.imsave("demo/pred_posnet.jpg",mel_output_posnet,cmap="hot")
plt.imsave("demo/align.jpg",align)
soundfile.write("demo/griff_ground_truth.wav",mel2audio(mel_spec),22050)
soundfile.write("demo/griff_pred.wav",mel2audio(mel_output),22050)
soundfile.write("demo/griff_pred_posnet.wav",mel2audio(mel_output_posnet),22050)

