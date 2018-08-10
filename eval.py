import argparse
import os
import re
from hparams import hparams, hparams_debug_string
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
  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
  name = 'eval-%d' % int(m.group(1)) if m else 'eval'
  return os.path.join(base_dir, name)


def run_eval(args):
  print(hparams_debug_string())
  synth = Synthesizer()
  synth.load(args.checkpoint)
  base_path = get_output_base_path(args.checkpoint)
  for i, text in enumerate(sentences):
    path = '%s-%d.wav' % (base_path, i)
    path_alignment = '%s-%d.png' % (base_path, i)
    print('Synthesizing: %s' % path)
    synth.synthesize(text, path, path_alignment)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.parse(args.hparams)
  run_eval(args)


if __name__ == '__main__':
  main()
