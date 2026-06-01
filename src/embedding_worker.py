"""
embedding_worker.py — Consumer: ai_processing_queue

Flow:
  1. Nhận payload từ ai_processing_queue
  2. GLiNER extract entities: skill, major, experience  (tách theo label)
  3. BGE-M3 encode 2 vector riêng biệt:
       • professional_embedding  ← skill + experience entities
       • major_embedding         ← major entities
  4. UPDATE job_postings SET professional_embedding, major_embedding, status='DONE'

Payload expected (từ standardization_job/main.py):
{
  "internal_job_id": "uuid",
  "text_for_ai": {
    "tags":         [...],   # list of strings
    "majors":       [...],   # list of strings (gợi ý major từ crawler)
    "description":  "...",
    "requirements": "...",
    "experience":   "..."
  },
  "location": [...]          # đính kèm nhưng KHÔNG dùng để tính vector
}
"""
import json
import logging
import pika
import numpy as np

from config import (
    RABBITMQ_CONN, AI_PROCESSING_QUEUE,
    GLINER_MODEL_NAME, BGE_MODEL_NAME, MODEL_CACHE_DIR,
)
from db import get_connection
from model_loader import load_gliner, load_bge

logger = logging.getLogger(__name__)

# ── GLiNER entity labels ────────────────────────────────────────────────────
GLINER_LABELS = ['skill', 'major', 'experience']


def build_raw_text(text_for_ai: dict) -> str:
    """
    Ghép tất cả các trường text thành một chuỗi duy nhất để đưa vào GLiNER.
    """
    parts = []

    # tags / majors (lists) — đưa vào để GLiNER có thêm context
    for field in ('tags', 'majors'):
        val = text_for_ai.get(field)
        if isinstance(val, list) and val:
            parts.append(' '.join(str(v) for v in val if v))
        elif isinstance(val, str) and val.strip():
            parts.append(val.strip())

    # text blobs
    for field in ('description', 'requirements', 'experience'):
        val = text_for_ai.get(field, '')
        if val and val.strip():
            parts.append(val.strip())

    return '\n'.join(parts)


def extract_entities_by_label(gliner_model, raw_text: str) -> dict:
    """
    Dùng GLiNER để extract entities, tách theo label:
      {
        'skill':      ['Python', 'Machine Learning', ...],
        'major':      ['Công nghệ thông tin', ...],
        'experience': ['2 năm kinh nghiệm', ...],
      }
    Fallback: nếu GLiNER không extract được gì, trả dict rỗng.
    """
    if not raw_text.strip():
        return {'skill': [], 'major': [], 'experience': []}

    # GLiNER có giới hạn ~512 tokens — cắt text nếu quá dài
    MAX_CHARS = 2000
    text = raw_text[:MAX_CHARS] if len(raw_text) > MAX_CHARS else raw_text

    entities = gliner_model.predict_entities(text, GLINER_LABELS, threshold=0.5)

    buckets: dict = {'skill': [], 'major': [], 'experience': []}
    seen: dict = {'skill': set(), 'major': set(), 'experience': set()}

    for ent in entities:
        label = ent.get('label', '').lower()
        token = ent.get('text', '').strip()
        if label in buckets and token and token.lower() not in seen[label]:
            seen[label].add(token.lower())
            buckets[label].append(token)

    return buckets


def generate_embedding(bge_model, clean_text: str) -> list | None:
    """
    Encode text bằng BGE-M3 → numpy array [1024] → Python list.
    Trả None nếu text rỗng.
    """
    if not clean_text or not clean_text.strip():
        return None

    output = bge_model.encode([clean_text], batch_size=1, max_length=512)
    # FlagEmbedding trả về dict với key 'dense_vecs'
    vec = output['dense_vecs'][0]
    return vec.tolist()


def save_embeddings_to_db(job_id: str, prof_embedding: list, major_embedding: list | None):
    """
    UPDATE job_postings SET professional_embedding, major_embedding, status='DONE'
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        prof_vec_str = '[' + ','.join(str(x) for x in prof_embedding) + ']'
        major_vec_str = (
            '[' + ','.join(str(x) for x in major_embedding) + ']'
            if major_embedding else None
        )
        cur.execute(
            """
            UPDATE job_postings
               SET professional_embedding = %s::vector,
                   major_embedding        = %s::vector,
                   status                 = 'DONE'
             WHERE id = %s
            """,
            (prof_vec_str, major_vec_str, job_id),
        )
        conn.commit()
        cur.close()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def process_message(gliner_model, bge_model, payload: dict):
    job_id      = payload.get('internal_job_id')
    text_for_ai = payload.get('text_for_ai', {})

    if not job_id:
        logger.warning('[EmbeddingWorker] Missing internal_job_id — skipping')
        return

    raw_text = build_raw_text(text_for_ai)

    # 1. GLiNER → tách entities theo label
    entities = extract_entities_by_label(gliner_model, raw_text)

    skill_exp_text = ', '.join(entities['skill'] + entities['experience'])
    major_text     = ', '.join(entities['major'])

    # Fallback nếu GLiNER không extract được skill/exp
    if not skill_exp_text:
        logger.warning(f'[EmbeddingWorker] No skill/experience entities for job {job_id} — using raw_text fallback')
        skill_exp_text = raw_text[:500]

    # 2. BGE encode 2 vectors riêng biệt
    prof_embedding  = generate_embedding(bge_model, skill_exp_text)
    major_embedding = generate_embedding(bge_model, major_text) if major_text else None

    if prof_embedding is None:
        logger.warning(f'[EmbeddingWorker] Empty professional text, skipping job {job_id}')
        return

    # 3. Lưu DB
    save_embeddings_to_db(job_id, prof_embedding, major_embedding)
    logger.info(
        f'[EmbeddingWorker] ✓ Job {job_id} → prof_emb({len(prof_embedding)}-dim) '
        f'| major_emb={"YES" if major_embedding else "NULL"} '
        f'| skills: {skill_exp_text[:60]}... '
        f'| major: {major_text[:40] or "(none)"}'
    )


def run(gliner_model, bge_model):
    """Start consuming ai_processing_queue."""
    params  = pika.URLParameters(RABBITMQ_CONN)
    params.heartbeat = 0  # Disable heartbeat timeout for long-running inference
    conn    = pika.BlockingConnection(params)
    channel = conn.channel()
    channel.queue_declare(queue=AI_PROCESSING_QUEUE, durable=True)
    channel.basic_qos(prefetch_count=1)

    def callback(ch, method, properties, body):
        try:
            payload = json.loads(body)
            process_message(gliner_model, bge_model, payload)
            if ch.is_open:
                ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            logger.error(f'[EmbeddingWorker] Error: {e}', exc_info=True)
            if ch.is_open:
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    channel.basic_consume(queue=AI_PROCESSING_QUEUE, on_message_callback=callback)
    logger.info(f'[EmbeddingWorker] Listening on {AI_PROCESSING_QUEUE} ...')
    channel.start_consuming()
