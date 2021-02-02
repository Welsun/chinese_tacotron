import tensorflow as tf
from text import symbols


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=1000,
        iters_per_checkpoint=1000,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=True,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=['embedding.weight'],

        ################################
        # Data Parameters             #
        ################################
        load_mel_from_disk=False,
        #wav_dir="dataset/Wave",
        wav_dir="audio_segments/wavs",
        # training_files='filelists/audio_text_train.txt',
        # validation_files='filelists/audio_text_test.txt',
        training_files='filelists/news_train.txt',
        validation_files='filelists/news_test.txt',
        text_cleaners=['basic_cleaners'],

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=1.0,  #32768.0,#max value of waveform,1 for biaobei
        sampling_rate=22050,#22050,# how many samples are in a second audio
        filter_length=1024,

        ## In paper, the window size is 50ms and hop size is 12.5ms
        ## 0.05*sampling_rate=0.05*22050=1102
        ## 0.0125*sr=275
        #这个地方和原文有些出入，感觉影响应该不是很大
        hop_length=256,#200,
        win_length=1024,#800,
        n_mel_channels=80,
        mel_fmin=55.0,##55 for male and 95 for female
        mel_fmax=8000.0,

        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),# vocabulary size
        symbols_embedding_dim=512,# embedding size

        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported,一次吐出几个frame
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=10000,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-4,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=16,
        mask_padding=True  # set model's padded outputs to padded values
    )

    if hparams_string:
        tf.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams
