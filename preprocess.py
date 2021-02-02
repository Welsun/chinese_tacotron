# 构建类似tacotron2 中的filelist 文件
import argparse
import os
from multiprocessing import cpu_count
from tqdm import tqdm
import shutil
from text import text_to_sequence
import hparams
from pinyin.parse_text_to_pyin import get_pyin

def get_all_file_lists(filelists,text_dir,all_file_lists):
    if not os.path.exists(filelists):
        os.mkdir(filelists)
    # else:
    #     return

    with open(text_dir,'r',encoding='utf-8') as f:
        i=1
        while i<=20000:
            title = f.readline().strip()
            audio_text=title.split("\t")
            audio=audio_text[0]+".wav"
            text=audio_text[-1]
            pinyin=f.readline().strip()
            # pinyin,txt=get_pyin(text)
            print(pinyin)
            content=audio+"|"+text
            #print(text_to_sequence(pinyin,cleaner_names=hp.text_cleaners))
            with open(os.path.join(filelists,all_file_lists),"a") as fw:
                fw.write(content+"\n")
            i+=2


if __name__ == '__main__':
    hp = hparams.create_hparams()
    wave_dir="./dataset/Wave"
    text_dir="./dataset/ProsodyLabeling/000001-010000.txt"
    filelists="filelists"
    all_file_lists="all_audio_text.txt"
    random_file_lists="random_all_file_lists.txt"
    train_filelist="bb_audio_text_train_filelist.txt"
    test_filelist = "bb_audio_text_test_filelist.txt"
    val_filelist = "bb_audio_text_val_filelist.txt"

    #step1: generate the all audio-text pairs
    get_all_file_lists(filelists,text_dir,all_file_lists)
    # step2: shuffle lists
    # In linux: shuf file1 -o file2

    #step3:
    # generate the train file lists





