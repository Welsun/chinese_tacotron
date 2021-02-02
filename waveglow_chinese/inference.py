# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import os
import sys
sys.path.append("..")
#sys.path.append(".")
from data_utils import TextMelLoader
from hparams import create_hparams


from text import text_to_sequence
from scipy.io.wavfile import write
import soundfile as sf
import torch
import numpy as np
from mel2samp import files_to_list, MAX_WAV_VALUE
from denoiser import Denoiser
from model import Tacotron2
from pinyin.parse_text_to_pyin import get_pyin
def load_model(hparams):
    model = Tacotron2(hparams).cuda()


    return model
def text2audio(waveglow_path,sigma,output_dir,sampling_rate,mel):
    waveglow = torch.load(waveglow_path)['model']
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.cuda().eval()

    denoiser = Denoiser(waveglow).cuda()

    with torch.no_grad():
        audio = waveglow.infer(mel.cuda(), sigma=sigma)
        # if denoiser_strength > 0:
        #     audio = denoiser(audio, denoiser_strength)
        #audio = audio * MAX_WAV_VALUE
    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    sf.write(os.path.join(output_dir,"pred2.wav"),audio,sampling_rate)

def main(mel_files, waveglow_path, sigma, output_dir, sampling_rate, is_fp16,
         denoiser_strength):
    mel_files = files_to_list(mel_files)
    waveglow = torch.load(waveglow_path)['model']
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.cuda().eval()
    if is_fp16:
        from apex import amp
        waveglow, _ = amp.initialize(waveglow, [], opt_level="O3")

    if denoiser_strength > 0:
        denoiser = Denoiser(waveglow).cuda()

    for i, file_path in enumerate(mel_files):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        mel = torch.load(file_path)
        mel = torch.autograd.Variable(mel.cuda())
        mel = torch.unsqueeze(mel, 0)
        mel = mel.half() if is_fp16 else mel
        with torch.no_grad():
            audio = waveglow.infer(mel, sigma=sigma)
            if denoiser_strength > 0:
                audio = denoiser(audio, denoiser_strength)
            audio = audio * MAX_WAV_VALUE
        audio = audio.squeeze()
        audio = audio.cpu().numpy()
        audio = audio.astype('int16')
        audio_path = os.path.join(
            output_dir, "{}_synthesis.wav".format(file_name))
        write(audio_path, sampling_rate, audio)
        print(audio_path)


if __name__ == "__main__":
    waveglow_path="./news_checkpoints/waveglow_64000"

    hp = create_hparams()
    hp.distributed_run = False
    # file = os.path.join("../"+hp.wav_dir, "000124.wav")
    # mel_loader = TextMelLoader("../"+hp.training_files, hp)
    # mel_spec = mel_loader.get_mel(file)

    checkpoint = '../news_output/checkpoint_90000'
    model = load_model(hp)
    model.load_state_dict(torch.load(checkpoint)['state_dict'])
    _ = model.cuda().eval().half()

    #text = "ka2 er2 pu3 pei2 wai4 sun1 wan2 hua2 ti1"
    # text="bu4 xiao3 xin1 nong4 huai4 lao3 ba4 de5 an4 mo2 yi3 lao3 ba4 dui4 wo3 shuo1 ni3 ruo4 an1 hao3 bian4 shi4 qing2 tian1"
    #text="zhong1 zhen4 tao1 yu2 nv3 er2 qian1 shou3 guang4 jie1"
    text="国务院总理李克强10月9日主持召开国务院常务会议，听取公立医院综合改革和医疗联合体建设进展情况汇报。"
    text, _ = get_pyin(text)
    sequence = np.array(text_to_sequence(text, ['basic_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence).cuda().long())
    mel_output, mel_output_posnet, _, alignment = model.inference(sequence)
    mel_output = mel_output.float().data.cpu()[0]
    mel_output_posnet = mel_output_posnet.float().data.cpu()[0]


    sigma=0.66
    text2audio(waveglow_path,sigma,"../demo/",22050,mel_output_posnet.unsqueeze(0))













    #waveglow = torch.load(waveglow_path)['model']
    # import argparse
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-f', "--filelist_path", required=True)
    # parser.add_argument('-w', '--waveglow_path',
    #                     help='Path to waveglow decoder checkpoint with model')
    # parser.add_argument('-o', "--output_dir", required=True)
    # parser.add_argument("-s", "--sigma", default=1.0, type=float)
    # parser.add_argument("--sampling_rate", default=22050, type=int)
    # parser.add_argument("--is_fp16", action="store_true")
    # parser.add_argument("-d", "--denoiser_strength", default=0.0, type=float,
    #                     help='Removes model bias. Start with 0.1 and adjust')
    #
    # args = parser.parse_args()
    #
    # main(args.filelist_path, args.waveglow_path, args.sigma, args.output_dir,
    #      args.sampling_rate, args.is_fp16, args.denoiser_strength)
