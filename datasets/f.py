from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
from util import audio
import re

_curly_re = re.compile(r'(.*?)(<.+?>)(.*)')
character_list = []

def build_from_path(in_dir, out_dir, num_workers, dataset, tqdm=lambda x: x):
  '''Preprocesses the LJ Speech dataset from a given input path into a given output directory.

    Args:
      in_dir: The directory where you have downloaded the LJ Speech dataset
      out_dir: The directory to write the output into
      num_workers: Optional number of worker processes to parallelize across
      tqdm: You can optionally pass tqdm to get a nice progress bar

    Returns:
      A list of tuples describing the training examples. This should be written to train.txt
  '''

  # We use ProcessPoolExecutor to parallize across processes. This is just an optimization and you
  # can omit it and just call _process_utterance on each input if you want.
  executor = ProcessPoolExecutor(max_workers=num_workers)
  futures = []
  index = 1

  wav_path = os.path.join(in_dir, 'wav')
  stm_path = os.path.join(in_dir, 'stm')

  for wav_file in os.listdir(wav_path):
      name_file = os.path.splitext(wav_file)[0]
      if not os.path.splitext(wav_file)[1] == '.wav':
          continue
      txt_file = '.'.join([name_file, 'stm'])
      wav_root = os.path.join(wav_path, wav_file)
      txt_root = os.path.join(stm_path, txt_file)

      with open(txt_root, 'r') as f:
          content = f.read()
      f.close()

      content = _curly_re.match(content)
      content = content.group(3)
      content = content.split(' ')
      content = ''.join(content)

      futures.append(executor.submit(partial(_process_utterance, out_dir, index, wav_root, content, dataset)))
      index += 1
      if index % 1000 == 0:
          print(index)
          print(content)
  return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, index, wav_path, text, dataset):
  '''Preprocesses a single utterance audio/text pair.

  This writes the mel and linear scale spectrograms to disk and returns a tuple to write
  to the train.txt file.

  Args:
    out_dir: The directory to write the spectrograms into
    index: The numeric index to use in the spectrogram filenames.
    wav_path: Path to the audio file containing the speech input
    text: The text spoken in the input audio file

  Returns:
    A (spectrogram_filename, mel_filename, n_frames, text) tuple to write to train.txt
  '''

  # Load the audio to a numpy array:
  wav = audio.load_wav(wav_path)

  # Compute the linear-scale spectrogram from the wav:
  spectrogram = audio.spectrogram(wav).astype(np.float32)
  n_frames = spectrogram.shape[1]

  # Compute a mel-scale spectrogram from the wav:
  mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

  # Write the spectrograms to disk:
  spectrogram_filename = 'f'+dataset+'-spec-%05d.npy' % index
  mel_filename = 'f'+dataset+'-mel-%05d.npy' % index
  #print(os.path.join(out_dir, spectrogram_filename))
  np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
  np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

  # Return a tuple describing this training example:
  return (spectrogram_filename, mel_filename, n_frames, text, wav_path)
