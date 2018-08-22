import argparse
import os
from datasets.wav_to_tfrecord import wav_to_tfrecord_read_from_text
from datasets.wav_to_npy import wav_to_npy_read_from_text
from hparams import hparams
import re, random



def preprocess_THCHS(args):

    pattern = '([A-Z0-9]+)\\_([0-9]+)\\.(wav)'
    input_path = os.path.join(args.data_path, 'data_thchs30/data_thchs30/data')
    data_name = 'THCHS'
    id_dict = {}
    ct = 0
    text_path = './datasets/name_THCHS_list.txt'

    with open(text_path, 'w') as f:
        for temp_filename in os.listdir(input_path):
            filename, extension = os.path.splitext(temp_filename)
            if not extension == '.wav':
                continue
            id = re.findall(pattern, temp_filename)[0][0]  # re.findall(pattern, wav_filename) = [('A7', '157', 'wav.trn')]
            if not id in id_dict.keys():
                id_dict[id] = ct
                ct += 1
            wav_path = os.path.join(input_path, filename + '.wav')
            trn_path = os.path.join(input_path, filename + '.wav.trn')
            with open(trn_path, 'r') as f1:
                text = f1.readline().strip()  # '时来运转 遇上 眼前 这位 知音 姑娘 还 因 工程 吃紧 屡 推 婚期\n'
                phone1 = f1.readline()  # 'zong3 er2 yan2 zhi1 wu2 lun4 na2 li3 ren2 chi1 yi4 wan3 she2 he2 mao1 huo4 zhe3 wa1 he2 shan4 yu2 yu2 xing4 fu2 de5 jia1 ting2 shi4 jue2 bu2 hui4 you3 sun3 shang1 de5\n'
                phone2 = f1.readline()  # 'z ong3 ee er2 ii ian2 zh ix1 uu u2 l un4 n a2 l i3 r en2 ch ix1 ii i4 uu uan3 sh e2 h e2 m ao1 h uo4 zh e3 uu ua1 h e2 sh an4 vv v2 vv v2 x ing4 f u2 d e5 j ia1 t ing2 sh ix4 j ve2 b u2 h ui4 ii iu3 s un3 sh ang1 d e5\n\n'
            #text2 = re.sub(' ', '', text.strip())  # '时来运转遇上眼前这位知音姑娘还因工程吃紧屡推婚期\n
            info_list = [wav_path, text, id_dict[id], phone2]
            f.write(str(info_list) + '\n')

    if args.data_type == 'tfrecord':
        # 写入tfrecord文件中
        wav_to_tfrecord_read_from_text(args=args, text_path=text_path, data_name=data_name, id_num=len(id_dict))
    if args.data_type == 'npy':
        # 写入npy文件中
        wav_to_npy_read_from_text(args=args, text_path=text_path, data_name=data_name, id_num=len(id_dict))


def preprocess_aishell(args):

    pattern = 'BAC009(S[0-9]+)W[0-9]+'
    input_path = args.data_path
    data_name = 'aishell'
    id_dict = {}
    ct = 0
    text_path = './datasets/name_aishell_list.txt'
    aishell_transcript_v08 = os.path.join(args.data_path, 'transcript', 'aishell_transcript_v0.8.txt')

    with open(text_path, 'w') as f:
        with open(aishell_transcript_v08, 'r') as f_in:
            for line in f_in:
                items = line.split(' ') # BAC009S0729W0485     提高  质量  是  教育  改革  发展  的  核心  任务
                id = re.findall(pattern, items[0])[0] # re.findall(pattern, items[0]) = ['S0729']

                if not id in id_dict.keys():
                    id_dict[id] = ct
                    ct += 1

                items = [x for x in items if x != '']

                wav_path = os.path.join(input_path, 'train', id, items[0]+'.wav') # '../data/data_aishell/train/S0729/BAC009S0729W0485.wav'
                text = ' '.join(items[1:]).strip() # 提高 质量 是 教育 改革 发展 的 核心 任务

                info_list = [wav_path, text, id_dict[id]]
                f.write(str(info_list) + '\n')


    if args.data_type == 'tfrecord':
        #写入tfrecord文件中
        wav_to_tfrecord_read_from_text(args=args, text_path=text_path, data_name=data_name, id_num=len(id_dict))
    if args.data_type == 'npy':
        # 写入npy文件中
        wav_to_npy_read_from_text(args=args, text_path=text_path, data_name=data_name, id_num=len(id_dict))



def preprocess_ljspeech(args):
    input_path = args.data_path
    data_name = 'ljspeech'
    ct = 0
    text_path = './datasets/name_ljspeech_list.txt'

    with open(os.path.join(input_path, 'metadata.csv'), encoding='utf-8') as f:

        with open(text_path, 'w') as f_write:
            for line in f:
                parts = line.strip().split('|')
                wav_path = os.path.join(input_path, 'wavs', '%s.wav' % parts[0])
                text = parts[2]

                info_list = [wav_path, text, ct]
                f_write.write(str(info_list) + '\n')

    if args.data_type == 'tfrecord':
        # 写入tfrecord文件中
        wav_to_tfrecord_read_from_text(args=args, text_path=text_path, data_name=data_name, id_num=1)
    if args.data_type == 'npy':
        # 写入npy文件中
        wav_to_npy_read_from_text(args=args, text_path=text_path, data_name=data_name, id_num=1)


def preprocess_vctk(args):
    input_path = args.data_path
    data_name = 'vctk'
    ct = 0
    text_path = './datasets/name_vctk_list.txt'

    vctk_path_wav = os.path.join(input_path, 'wav48')
    vctk_path_txt = os.path.join(input_path, 'txt')

    with open('./datasets/name_vctk_list.txt', 'w') as fvctk:
        for dir_name in os.listdir(vctk_path_wav):
            # dir_name = 'p300'
            # wav_dir  = '/home/pattern/songjinming/tts/data/vctk-Corpus/wav48/p300'
            # txt_dir  = '/home/pattern/songjinming/tts/data/vctk-Corpus/txt/p300'
            wav_dir = os.path.join(vctk_path_wav, dir_name)
            txt_dir = os.path.join(vctk_path_txt, dir_name)
            for wav_file in os.listdir(wav_dir):
                # wav_file  = 'p300_224.wav'
                # name_file = 'p300_224'
                # txt_file  = 'p300_224.txt'
                # wav_root  = '/home/pattern/songjinming/tts/data/vctk-Corpus/wav48/p300/p300_224.wav'
                # txt_root  = '/home/pattern/songjinming/tts/data/vctk-Corpus/txt/p300/p300_224.txt'
                name_file = os.path.splitext(wav_file)[0]
                # some file is not wav file and just skip
                # print(os.path.splitext(wav_file))
                if not os.path.splitext(wav_file)[1] == '.wav':
                    continue
                txt_file = '.'.join([name_file, 'txt'])
                wav_root = os.path.join(wav_dir, wav_file)
                txt_root = os.path.join(txt_dir, txt_file)
                # txt
                # some wav files dont have correspond txt file
                try:
                    text = open(txt_root, 'r').read()
                    text = re.sub('\n', '', text)
                except:
                    continue
                # write
                info_list = [wav_root, text, ct]
                fvctk.write(str(info_list))
                fvctk.write('\n')
            ct += 1

    if args.data_type == 'tfrecord':
        # 写入tfrecord文件中
        wav_to_tfrecord_read_from_text(args=args, text_path=text_path, data_name=data_name, id_num=ct)
    if args.data_type == 'npy':
        # 写入npy文件中
        wav_to_npy_read_from_text(args=args, text_path=text_path, data_name=data_name, id_num=ct)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, default=None)
    parser.add_argument('--data_path', required=True, help='input data path')
    parser.add_argument('--dataset', required=True, choices=['THCHS', 'aishell', 'ljspeech', 'vctk'])
    parser.add_argument('--num_workers', type=int, default=30)
    parser.add_argument('--data_type', type=str, default='npy', help='the data type to be saved, could be npy or tfrecord')
    args = parser.parse_args()
    if args.dataset == 'THCHS':
        preprocess_THCHS(args)
    if args.dataset == 'aishell':
        preprocess_aishell(args)
    if args.dataset == 'ljspeech':
        preprocess_ljspeech(args)
    if args.dataset == 'vctk':
        preprocess_vctk(args)


if __name__ == "__main__":
    main()
'''
recommended command
python3 preprocess_data.py --data_path ../data/THCHS/ --dataset THCHS --output /ssd1 --data_type npy
python3 preprocess_data.py --data_path ../data/data_aishell/ --dataset aishell --output /ssd1 --data_type npy
python3 preprocess_data.py --data_path ../data/LJSpeech-1.0/ --dataset ljspeech --output /ssd1 --data_type npy
python3 preprocess_data.py --data_path ../data/vctk-Corpus/ --dataset vctk --output /ssd1 --data_type npy

'''