"""
main.py — AI Service entry point.
Tải models một lần duy nhất rồi khởi động 2 workers trong 2 threads:
  - embedding_worker : ai_processing_queue  → GLiNER + BGE-M3 → DB
  - matching_worker  : match_request_queue  → cosine sim → match_results
"""
import sys
import logging
import threading

# ── Logging setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger('ai_service')

# ── Import config FIRST (sets HF_TOKEN env vars) ────────────────────────────
from config import GLINER_MODEL_NAME, BGE_MODEL_NAME, MODEL_CACHE_DIR
from model_loader import load_gliner, load_bge
import embedding_worker
import matching_worker
import student_embedding_worker


def main():
    logger.info('=' * 60)
    logger.info('  AI Service — Starting')
    logger.info(f'  GLiNER model : {GLINER_MODEL_NAME}')
    logger.info(f'  BGE-M3 model : {BGE_MODEL_NAME}')
    logger.info(f'  Cache dir    : {MODEL_CACHE_DIR}')
    logger.info('=' * 60)

    # ── Load models ONCE at startup ──────────────────────────────────────────
    # Both workers share the same model instances (read-only inference = thread-safe)
    logger.info('[Startup] Loading GLiNER model...')
    gliner_model = load_gliner(GLINER_MODEL_NAME, MODEL_CACHE_DIR)

    logger.info('[Startup] Loading BGE-M3 model...')
    bge_model = load_bge(BGE_MODEL_NAME, MODEL_CACHE_DIR)

    logger.info('[Startup] ✓ Both models loaded. Starting workers...')

    # ── Start Job embedding worker in background thread ──────────────────────
    embed_thread = threading.Thread(
        target=embedding_worker.run,
        args=(gliner_model, bge_model),
        name='JobEmbeddingWorker',
        daemon=True,
    )
    embed_thread.start()
    logger.info('[Startup] JobEmbeddingWorker thread started.')

    # ── Start Student embedding worker in background thread ──────────────────
    student_embed_thread = threading.Thread(
        target=student_embedding_worker.run,
        args=(bge_model,), # Chỉ cần BGE-M3 vì text payload đã được backend build sẵn
        name='StudentEmbeddingWorker',
        daemon=True,
    )
    student_embed_thread.start()
    logger.info('[Startup] StudentEmbeddingWorker thread started.')

    # ── Start matching worker in main thread ─────────────────────────────────
    # (blocking — keeps process alive)
    logger.info('[Startup] Starting MatchingWorker in main thread...')
    matching_worker.run()


if __name__ == '__main__':
    main()
