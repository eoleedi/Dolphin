# encoding: utf8

import yaml
import tqdm
import pydub
import logging
import hashlib
import os.path
import argparse
import numpy as np
from pathlib import Path
from argparse import Namespace
from os.path import dirname, join, abspath, join
from distutils.util import strtobool
from typing import Union, Optional, Tuple, List

import torch
import modelscope
from modelscope.models.audio.funasr.model import GenericFunASR

from .audio import load_audio, convert_audio
from .model import DolphinSpeech2Text, TranscribeResult
from .languages import LANGUAGE_REGION_CODES, LANGUAGE_CODES
from .constants import SPEECH_LENGTH

logger = logging.getLogger("dolphin")

VAD_MODEL = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"


MODELS = {
    "base": {
        "model_id": "DataoceanAI/dolphin-base",
        "download_url": "http://so-algorithm-prod.oss-cn-beijing.aliyuncs.com/models/dolphin/base.pt",
        "sha256": "688f0cdb26da2684a4eec200a432091920287585e8e332507cbe9c1ab6d77401",
        "config": {
            "encoder": {
                "output_size": 512,
                "attention_heads": 8,
                "cgmlp_linear_units": 2048,
                "num_blocks": 6,
                "linear_units": 2048,
            },
            "decoder": {
                "attention_heads": 8,
                "linear_units": 2048,
                "num_blocks": 6,
            }
        }
    },
    "small": {
        "model_id": "DataoceanAI/dolphin-small",
        "download_url": "http://so-algorithm-prod.oss-cn-beijing.aliyuncs.com/models/dolphin/small.pt",
        "sha256": "e5a52b9a713d294d5a2d929f5e7f6a18d951a8155ede80f935a74b76b0432b17",
        "config": {
            "encoder": {
                "output_size": 768,
                "attention_heads": 12,
                "cgmlp_linear_units": 3072,
                "num_blocks": 12,
                "linear_units": 1536,
            },
            "decoder": {
                "attention_heads": 12,
                "linear_units": 3072,
                "num_blocks": 12,
            }
        }
    },
}


def str2bool(value: str) -> bool:

    return bool(strtobool(value))


def parser_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=str, help="audio file path")
    parser.add_argument("--model", type=str, default="small", help="model name (default: small)")
    parser.add_argument("--model_dir", type=Path, default=None, help="model checkpoint download diretory")
    parser.add_argument("--lang_sym", type=str, default=None, help="language symbol (e.g. zh)")
    parser.add_argument("--region_sym", type=str, default=None, help="regiion symbol (e.g. CN)")
    parser.add_argument("--device", type=str, default=None, help="torch device (default: None)")
    parser.add_argument("--normalize_length", type=str2bool, default=False, help="whether to normalize length (default: false)")
    parser.add_argument("--padding_speech", type=str2bool, default=False, help="whether padding speech to 30 seconds (default: false)")
    parser.add_argument("--predict_time", type=str2bool, default=True, help="whether predict timestamp (default: true)")
    parser.add_argument("--beam_size", type=int, default=5, help="number of beams in beam search (default: 5)")
    parser.add_argument("--maxlenratio", type=float, default=0.0, help="Input length ratio to obtain max output length (default: 0.0)")

    args = parser.parse_args()
    return args


def detect_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"

    return device


def load_model(
    model_name: str,
    model_dir: Union[Path, str],
    device: Optional[Union[str, torch.device]] = None,
    **kwargs
) -> DolphinSpeech2Text:
    """
    Load DolphinSpeech2Text model.

    Args:
        model_name: model name (e.g. small)
        model_dir: model download directory
        device: the pytorch device

    Returns:
        DolphinSpeech2Text instance
    """
    if device is None:
        device = detect_device()
        logger.info(f"auto detect device: {device}")

    model_config = MODELS[model_name]["config"]
    train_cfg_file = join(dirname(abspath(__file__)), "assets/config.yaml")
    with open(train_cfg_file, "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)
        train_cfg["encoder_conf"].update(**model_config["encoder"])
        train_cfg["decoder_conf"].update(**model_config["decoder"])

    if isinstance(model_dir, str):
        model_dir = Path(model_dir)

    model_file = model_dir / f"{model_name}.pt"
    download_model = True
    if model_file.exists():
        with open(model_file, "rb") as f:
            model_bytes = f.read()
        if hashlib.sha256(model_bytes).hexdigest() == MODELS[model_name]["sha256"]:
            download_model = False
        else:
            model_file.unlink(missing_ok=True)
            logger.warning("model SHA256 chechsum mismatch, redownload model...")

    if download_model:
        # Download model
        model_dir.mkdir(exist_ok=True)
        _download_from_modelscope(
            model_id=MODELS[model_name]["model_id"],
            local_dir=model_dir,
            allow_file_pattern=f"{model_name}.pt",
        )

    model = DolphinSpeech2Text(
        s2t_train_config=train_cfg,
        s2t_model_file=model_file,
        device=device,
        task_sym=kwargs.get("task_sym", "<asr>"),
        predict_time=kwargs.get("predict_time", True),
        **kwargs,
    )
    return model


def _download_from_modelscope(model_id: str, local_dir: str, allow_file_pattern: str):
    modelscope.snapshot_download(
        model_id=model_id,
        local_dir=local_dir,
        allow_file_pattern=allow_file_pattern,
        repo_type="model",
    )


def validate_lang_region(lang_sym: str, region_sym: str):

    if all([lang_sym, region_sym]):
        if f"{lang_sym}-{region_sym}" not in LANGUAGE_REGION_CODES:
            raise Exception("Unsupport language or region!")
    elif lang_sym and region_sym is None:
        if lang_sym not in LANGUAGE_CODES:
            raise Exception("Unsupport language!")

    return True


def transcribe_long(
        audio: str,
        device: str,
        model: str,
        model_dir: str,
        lang_sym: str,
        region_sym: str,
        **kwargs,
    ) -> List[TranscribeResult]:
    """
    Transcribe audio to text.

    Args:
        audio: audio path
        device: inference device, (e.g. cpu, cuda, mps)
        model: model name (e.g. small)
        model_dir: model download directory
        lang_sym: language symbol (e.g. zh)
        region_sym: regiion symbol (e.g. CN)

    Returns:
        List[TranscribeResult]
    """
    if model not in MODELS:
        logging.error(f"Unknown model {model}, Dolphin open source base, small model, please config the correct model.")
        return

    validate_lang_region(lang_sym, region_sym)

    model_dir = model_dir if model_dir else os.path.expanduser("~/.cache/dolphin")
    model_dir = Path(model_dir)

    logging.info("download vad model...")
    vad_model_dir = Path(os.path.expanduser("~/.cache/dolphin/speech_fsmn_vad"))
    vad_model_dir.mkdir(exist_ok=True)
    _download_from_modelscope(VAD_MODEL, vad_model_dir, None)

    logging.info("loading ")
    vad_model = GenericFunASR(vad_model_dir, max_single_segment_time=SPEECH_LENGTH*1000, device="cpu")
    segments = vad_model(input=audio, disable_pbar=True)[0]["value"]

    logger.info("loading model...")
    model_kwargs = {
        "device": device,
        "normalize_length": kwargs.get("normalize_length", False),
        "beam_size": kwargs.get("beam_size", 5),
        "maxlenratio": kwargs.get("maxlenratio", 0.0),
    }

    model = load_model(model, model_dir, **model_kwargs)

    results = []
    logger.info("inference...")
    tmp_audio = f"{audio}.wav"
    convert_audio(audio, tmp_audio)
    audio_segment = pydub.AudioSegment.from_wav(tmp_audio)

    for seg in segments:
        s, e = seg
        raw_data = audio_segment[s:e].raw_data
        waveform = np.frombuffer(raw_data, np.int16).flatten().astype(np.float32) / 32768.0
        result = model(
            speech=waveform,
            lang_sym=lang_sym,
            region_sym=region_sym,
            predict_time=kwargs.get("predict_time", True),
            padding_speech=kwargs.get("padding_speech", False)
        )

        st = round(s / 1000, 2)
        et = round(e / 1000, 2)
        logger.info(f"segment: {st} - {et}, lang: {result.language}, region: {result.region}, text: {result.text_nospecial}")
        results.append(result)

    # clean tmp audio file
    Path(tmp_audio).unlink(missing_ok=True)

    return result


def transcribe(
        audio: str,
        device: str,
        model: str,
        model_dir: str,
        lang_sym: str,
        region_sym: str,
        **kwargs,
    ) -> TranscribeResult:
    """
    Transcribe audio to text.

    Args:
        audio: audio path
        device: inference device, (e.g. cpu, cuda, mps)
        model: model name (e.g. small)
        model_dir: model download directory
        lang_sym: language symbol (e.g. zh)
        region_sym: regiion symbol (e.g. CN)

    Returns:
        TranscribeResult
    """
    if model not in MODELS:
        logging.error(f"Unknown model {model}, Dolphin open source base, small model, please config the correct model.")
        return

    validate_lang_region(lang_sym, region_sym)

    model_dir = model_dir if model_dir else os.path.expanduser("~/.cache/dolphin")
    model_dir = Path(model_dir)

    logger.info("loading model...")
    model_kwargs = {
        "device": device,
        "normalize_length": kwargs.get("normalize_length", False),
        "beam_size": kwargs.get("beam_size", 5),
        "maxlenratio": kwargs.get("maxlenratio", 0.0),
    }

    model = load_model(model, model_dir, **model_kwargs)
    waveform = load_audio(audio)

    logger.info("inference...")
    result = model(
        speech=waveform,
        lang_sym=lang_sym,
        region_sym=region_sym,
        predict_time=kwargs.get("predict_time", True),
        padding_speech=kwargs.get("padding_speech", False)
    )

    logger.info(f"decode result, rtf: {result.rtf}, language: {result.language}, region: {result.region}, text: {result.text}")
    return result


def cli():
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    LOGGING_FORMAT="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)

    # filter framework interanl logs
    logging.getLogger("espnet").setLevel(logging.ERROR)
    logging.getLogger("root").setLevel(logging.ERROR)
    logging.getLogger("dolphin").setLevel(logging.INFO)

    args = parser_args()

    audio_duration = pydub.AudioSegment.from_file(args.audio).duration_seconds
    transcribe_fn = transcribe_long if audio_duration > SPEECH_LENGTH else transcribe
    transcribe_fn(**vars(args))


if __name__ == "__main__":
    cli()
