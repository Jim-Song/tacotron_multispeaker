import argparse
import os
import re
from hparams import hparams, hparams_debug_string
from tensorflow.python.training.saver import get_checkpoint_state
from synthesizer import Synthesizer


sentences = ['习近平多次强调',
             '安全是上海合作组织发展的前提',
             '务实合作是组织发展的原动力',
             '人文交流是组织发展的民意基础和社会基础',
             '安全、经济、人文三大领域',
             '上合组织各个成员国在政治上不断增强互信',
             '在经济上不断深化务实合作',
             '在民间交流中人文纽带越拉越紧',
             '逐步形成了安全',
             '经济与人文并重',
             '官方与民间并举的全面合作机制',
             '这是地区一体化进程的必然趋势',
             '也是上海合作组织未来发展的大方向',
             '直砍少年顶门',
             '那少年避向右侧',
             '左手剑一引',
             '青钢剑疾刺那汉子大腿',
             '两人剑法迅捷',
             '全力相搏',
             '练武厅东坐着二人',
             '上首是个四十左右的中年道姑']
'''
    # From July 8, 2017 New York Times:
    'Scientists at the CERN laboratory say they have discovered a new particle.',
    'There’s a way to measure the acute emotional intelligence that has never gone out of style.',
    'President Trump met with other leaders at the Group of 20 conference.',
    'The Senate\'s bill to repeal and replace the Affordable Care Act is now imperiled.',
    # From Google's Tacotron example page:
    'Generative adversarial network or variational auto-encoder.',
    'The buses aren\'t the problem, they actually provide a solution.',
    'Does the quick brown fox jump over the lazy dog?',
    'Talib Kweli confirmed to AllHipHop that he will be releasing an album in the next year.',
'''


def get_output_base_path(checkpoint_path):
    base_dir = os.path.dirname(checkpoint_path)
    os.makedirs(os.path.join(base_dir, 'eval'), exist_ok=True)
    m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
    name = 'eval-%d' % int(m.group(1)) if m else 'eval'
    return os.path.join(base_dir, 'eval', name)


def run_eval(args):
    if not args.ckpt_path:
        run_name = args.name or args.model
        log_dir = os.path.join(args.base_dir, 'logs-%s-%s' % (run_name, args.description))
        print("Trying to restore saved checkpoints from {} ...".format(log_dir))
        ckpt = get_checkpoint_state(log_dir)
        if ckpt:
            print("Checkpoint found: {}".format(ckpt.model_checkpoint_path))
            ckpt_path = ckpt.model_checkpoint_path
        else:
            print('no model found')
            raise
    else:
        ckpt_path = args.ckpt_path
    print(hparams_debug_string())
    synth = Synthesizer()
    synth.load(ckpt_path)
    base_path = get_output_base_path(ckpt_path)
    for i, text in enumerate(sentences):
        path = '%s-%d-identity-%d-%s.wav' % (base_path, i, args.identity, text)
        path_alignment = '%s-%d-identity-%d.png' % (base_path, i, args.identity)
        print('Synthesizing: %s' % path)
        synth.synthesize(text, args.identity, path, path_alignment)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./logs/')
    parser.add_argument('--model', default='tacotron')
    parser.add_argument('--name', help='Name of the run. Used for logging. Defaults to model name.')
    parser.add_argument('--hparams', default='', help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--ckpt_path', default=None, help='the model to be restored')
    parser.add_argument('--description', default=None, help='the model to be restored')
    parser.add_argument('--identity', default=0, type=int, help="the person's speech to be synthesized")

    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    hparams.parse(args.hparams)
    
    run_eval(args)


if __name__ == '__main__':
    main()
