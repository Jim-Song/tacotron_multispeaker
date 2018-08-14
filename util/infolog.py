from datetime import datetime


_format = '%Y-%m-%d %H:%M:%S.%f'
_file = None


def init(filename):
  global _file
  _close_logfile()
  _file = open(filename, 'a')
  _file.write('\n-----------------------------------------------------------------\n')
  _file.write('Starting new training run\n')
  _file.write('-----------------------------------------------------------------\n')


def log(msg):
  print(msg)
  if _file is not None:
    _file.write('[%s]  %s\n' % (datetime.now().strftime(_format)[:-3], msg))


def _close_logfile():
  global _file
  if _file is not None:
    _file.close()
    _file = None


