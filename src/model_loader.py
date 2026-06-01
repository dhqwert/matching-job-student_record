"""
model_loader.py — Singleton loader cho GLiNER và BGE-M3.
Models được tải một lần duy nhất khi service khởi động,
lưu vào MODEL_CACHE_DIR, không tải lại khi restart (nếu đã cache).
"""
import os
import sys
import logging

logger = logging.getLogger(__name__)

_gliner_model = None
_bge_model    = None


def load_gliner(model_name: str, cache_dir: str):
    """
    Tải GLiNER model. Lần đầu → download + lưu cache_dir.
    Các lần sau → đọc trực tiếp từ cache_dir.
    """
    global _gliner_model
    if _gliner_model is not None:
        return _gliner_model

    from gliner import GLiNER
    os.makedirs(cache_dir, exist_ok=True)

    logger.info(f"[GLiNER] Loading model '{model_name}' (cache: {cache_dir}) ...")
    _gliner_model = GLiNER.from_pretrained(model_name, cache_dir=cache_dir)
    logger.info("[GLiNER] Model ready.")
    return _gliner_model


def load_bge(model_name: str, cache_dir: str):
    """
    Tải BGE-M3 model qua FlagEmbedding. Lần đầu → download + cache.
    Các lần sau → dùng lại từ cache.
    """
    global _bge_model
    if _bge_model is not None:
        return _bge_model

    from FlagEmbedding import BGEM3FlagModel
    os.makedirs(cache_dir, exist_ok=True)

    logger.info(f"[BGE-M3] Loading model '{model_name}' (cache: {cache_dir}) ...")
    _bge_model = BGEM3FlagModel(model_name, use_fp16=True, cache_dir=cache_dir)
    logger.info("[BGE-M3] Model ready.")
    return _bge_model
