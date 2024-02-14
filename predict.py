# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import json
import librosa
import shutil
import torch
import soundfile as sf
from cog import BasePredictor, Input, Path
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchMultiTaskAED
from nemo.collections.asr.parts.utils.transcribe_utils import (
    get_buffered_pred_feat_multitaskAED,
)

# Enable faster download speed
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


# map src_lang and tgt_lang from long versions to short
LANG_LONG_TO_LANG_SHORT = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
}

SAMPLE_RATE = 16000  # Hz
MAX_AUDIO_MINUTES = 20  # wont try to transcribe if longer than this


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = ASRModel.from_pretrained("nvidia/canary-1b")
        self.model.eval()

        # make sure beam size always 1 for consistency
        self.model.change_decoding_strategy(None)
        decoding_cfg = self.model.cfg.decoding
        decoding_cfg.beam.beam_size = 1
        self.model.change_decoding_strategy(decoding_cfg)

        # setup for buffered inference
        self.model.cfg.preprocessor.dither = 0.0
        self.model.cfg.preprocessor.pad_to = 0

        feature_stride = self.model.cfg.preprocessor["window_stride"]
        self.model_stride_in_secs = (
            feature_stride * 8
        )  # 8 = model stride, which is 8 for FastConformer

        self.frame_asr = FrameBatchMultiTaskAED(
            asr_model=self.model,
            frame_len=40.0,
            total_buffer=40.0,
            batch_size=16,
        )

        self.amp_dtype = torch.float16

    def predict(
        self,
        audio: Path = Input(description="Input audio file. Audio file more than 20 minutes will be truncated to trascribe the first 20 minutes."),
        audio_language: str = Input(
            description="Language of the input audio.",
            choices=["English", "Spanish", "French", "German"],
            default="English",
        ),
        tgt_language: str = Input(
            description="Language of the transcription.",
            choices=["English", "Spanish", "French", "German"],
            default="English",
        ),
        pnc: bool = Input(
            description="Punctuation & Capitalization in transcript.", default=True
        ),
    ) -> str:
        """Run a single prediction on the model"""

        exp_dir = "exp_temp"
        if os.path.exists(exp_dir):
            shutil.rmtree(exp_dir)
        os.makedirs(exp_dir)

        converted_audio_filepath = os.path.join(exp_dir, "converted.wav")
        duration = convert_audio(str(audio), converted_audio_filepath)

        src_lang = LANG_LONG_TO_LANG_SHORT[audio_language]
        tgt_lang = LANG_LONG_TO_LANG_SHORT[tgt_language]

        # infer taskname from src_lang and tgt_lang
        taskname = "asr" if src_lang == tgt_lang else "s2t_translation"
        # update pnc variable to be "yes" or "no"
        pnc = "yes" if pnc else "no"

        # make manifest file and save
        manifest_data = {
            "audio_filepath": converted_audio_filepath,
            "source_lang": src_lang,
            "target_lang": tgt_lang,
            "taskname": taskname,
            "pnc": pnc,
            "answer": "predict",
            "duration": str(duration),
        }

        manifest_filepath = os.path.join(exp_dir, "manifest.json")

        with open(manifest_filepath, "w") as fout:
            line = json.dumps(manifest_data)
            fout.write(line + "\n")

        # call transcribe, passing in manifest filepath
        if duration < 40:
            output_text = self.model.transcribe(manifest_filepath)[0]
        else:  # do buffered inference
            with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                with torch.no_grad():
                    hyps = get_buffered_pred_feat_multitaskAED(
                        self.frame_asr,
                        self.model.cfg.preprocessor,
                        self.model_stride_in_secs,
                        self.model.device,
                        manifest=manifest_filepath,
                        filepaths=None,
                    )

                    output_text = hyps[0].text

        return output_text


def convert_audio(audio_filepath, out_filename):
    """
    Convert all files to monochannel 16 kHz wav files.
    Do not convert and raise error if audio too long.
    Returns output filename and duration.
    """

    data, sr = librosa.load(audio_filepath, sr=None, mono=True)

    duration = librosa.get_duration(y=data, sr=sr)

    if duration / 60.0 > MAX_AUDIO_MINUTES:
        # Calculate the number of samples to keep
        max_samples = int(MAX_AUDIO_MINUTES * 60 * sr)
        data = data[:max_samples]  # Trim the data to the maximum allowed duration
        duration = MAX_AUDIO_MINUTES * 60  # Update duration to the maximum allowed

    if sr != SAMPLE_RATE:
        data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)
    # save output audio
    sf.write(out_filename, data, SAMPLE_RATE)

    return duration
