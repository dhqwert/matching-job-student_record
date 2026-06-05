"""
model_loader.py — Singleton loader cho GLiNER và BGE-M3.
Models được tải một lần duy nhất khi service khởi động,
lưu vào MODEL_CACHE_DIR, không tải lại khi restart (nếu đã cache).
"""
import os
import sys
import logging

logger = logging.getLogger(__name__)

_bge_model    = None


def load_bge(model_name: str, cache_dir: str):
    """
    Tải BGE-M3 model qua FlagEmbedding. Lần đầu → download + cache.
    Các lần sau → dùng lại từ cache.
    """
    global _bge_model
    if _bge_model is not None:
        return _bge_model

    import os
    
    # Check if MinIO is configured for BGE (DO THIS BEFORE IMPORTING HEAVY LIBS)
    MINIO_BUCKET = os.getenv("MINIO_BUCKET")
    BGE_MINIO_PREFIX = os.getenv("BGE_MINIO_PREFIX", "")
    
    if MINIO_BUCKET:
        from s3_downloader import download_model_from_minio
        local_dir_name = BGE_MINIO_PREFIX.replace('/', '_') if BGE_MINIO_PREFIX else MINIO_BUCKET
        local_bge_dir = os.path.join(cache_dir, local_dir_name)
        success = download_model_from_minio(MINIO_BUCKET, BGE_MINIO_PREFIX, local_bge_dir)
        if success:
            model_name = local_bge_dir

    from FlagEmbedding import BGEM3FlagModel
    os.makedirs(cache_dir, exist_ok=True)

    logger.info(f"[BGE-M3] Loading model '{model_name}' (cache: {cache_dir}) ...")
    _bge_model = BGEM3FlagModel(model_name, use_fp16=True, cache_dir=cache_dir)
    logger.info("[BGE-M3] Model ready.")
    return _bge_model
