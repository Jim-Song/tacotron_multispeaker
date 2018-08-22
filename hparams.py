import tensorflow as tf


# Default hyperparameters:
hparams = tf.contrib.training.HParams(
    # Comma-separated list of cleaners to run on text prior to training and eval. For non-English
    # text, you may want to use "basic_cleaners" or "transliteration_cleaners" See TRAINING_DATA.md.
    cleaners='english_cleaners',

    # Audio:
    num_mels=80,
    num_freq=1025,
    sample_rate=20000,
    frame_length_ms=50,
    frame_shift_ms=12.5,
    preemphasis=0.97,
    min_level_db=-100,
    ref_level_db=20,

    # Model:
    # TODO: add more configurable hparams
    outputs_per_step=1, # # #

    # Training:
    batch_size=32, # # #
    adam_beta1=0.9,
    adam_beta2=0.999,
    initial_learning_rate=0.002,#0.002
    decay_learning_rate=True,
    use_phone_input=False,  # # #   # Use CMUDict during training to learn pronunciation of ARPAbet phonemes
    per_cen_phone_input=0.0,  # # # #range from 0 to 1, 0--no phone input, 1--all phone input

    # Eval:
    max_iters=40000,
    griffin_lim_iters=100,
    power=1.5,                            # Power to raise magnitudes to prior to Griffin-Lim

    # network settings
    embedding_text_channels=256,
    embedding_id_channels=64,

    # input
    bucket_len=1,
    eos=True,

    #regularity
    overwrought=0.0, # 0.0001
    oneorder_dynamic=0.0, # 0.00002
    variance_between_row=0.0,
    alignment_entropy=0.0, # 1 whether apply entropy on alignment


)


def hparams_debug_string():
    values = hparams.values()
    hp = ['    %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
















