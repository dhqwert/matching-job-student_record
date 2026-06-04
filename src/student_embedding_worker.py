"""
student_embedding_worker.py — Consumer: student_embedding_queue

Flow:
  1. Nhận payload từ student_embedding_queue (chứa text_payload đã được build sẵn từ backend).
  2. BGE-M3 encode text_payload → vector[1024].
  3. UPDATE student_records SET professional_embedding = vector WHERE student_code = ...
"""
import json
import logging
import pika

import requests
from config import RABBITMQ_CONN, STUDENT_EMBEDDING_QUEUE, BACKEND_API_URL, AI_API_KEY
from db import get_connection

logger = logging.getLogger(__name__)


def generate_embedding(bge_model, clean_text: str) -> list:
    """
    Encode text bằng BGE-M3 → numpy array [1024] → Python list.
    """
    if not clean_text or not clean_text.strip():
        return None

    # Max length for BGE-M3 is 8192, but typically we keep it under 512 for fast processing
    output = bge_model.encode([clean_text], batch_size=1, max_length=512)
    vec = output['dense_vecs'][0]
    return vec.tolist()


def save_student_embedding_to_db(student_code: str, embedding: list, major_embedding: list = None):
    """
    UPDATE student_records SET professional_embedding = <vector>
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        vec_str = '[' + ','.join(str(x) for x in embedding) + ']'
        major_vec_str = None
        if major_embedding:
            major_vec_str = '[' + ','.join(str(x) for x in major_embedding) + ']'
            
        cur.execute(
            """
            UPDATE student_records
               SET professional_embedding = %s::vector, major_embedding = %s::vector
             WHERE student_code = %s
            """,
            (vec_str, major_vec_str, student_code),
        )
        conn.commit()
        cur.close()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


import re

def process_message(bge_model, payload: dict):
    student_code = payload.get('student_code')

    if not student_code:
        logger.warning('[StudentEmbeddingWorker] Missing student_code — skipping')
        return

    # Fetch clean profile from Backend API
    try:
        url = f"{BACKEND_API_URL}/ai/student-records/{student_code}"
        headers = {}
        if AI_API_KEY:
            headers['x-api-key'] = AI_API_KEY
            
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json().get('data', {})
        career_profile = data.get('career_profile', {})
    except Exception as e:
        logger.error(f"[StudentEmbeddingWorker] Failed to fetch profile for {student_code}: {e}")
        return

    # Extract major text and professional text
    major_list = career_profile.get('major', [])
    major_text = ", ".join(major_list) if major_list else "Không xác định"

    skill_list = career_profile.get('skill', [])
    exp_list = career_profile.get('experience', [])
    prof_text = ", ".join(skill_list) + " " + " ".join(exp_list)
    if not prof_text.strip():
        prof_text = "Chưa có thông tin kỹ năng và kinh nghiệm"

    major_embedding = generate_embedding(bge_model, major_text)
    embedding = generate_embedding(bge_model, prof_text)

    if embedding is None:
        logger.warning(f'[StudentEmbeddingWorker] Failed to generate embedding for student {student_code}')
        return

    save_student_embedding_to_db(student_code, embedding, major_embedding)
    logger.info(f'[StudentEmbeddingWorker] ✓ Student {student_code} → embedded (Prof: {len(embedding)}-dim, Major: {len(major_embedding)}-dim)')


def run(bge_model):
    """Start consuming student_embedding_queue."""
    params  = pika.URLParameters(RABBITMQ_CONN)
    params.heartbeat = 0
    conn    = pika.BlockingConnection(params)
    channel = conn.channel()
    channel.queue_declare(queue=STUDENT_EMBEDDING_QUEUE, durable=True)
    channel.basic_qos(prefetch_count=1)

    def callback(ch, method, properties, body):
        try:
            payload = json.loads(body)
            process_message(bge_model, payload)
            if ch.is_open:
                ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            logger.error(f'[StudentEmbeddingWorker] Error: {e}', exc_info=True)
            if ch.is_open:
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    channel.basic_consume(queue=STUDENT_EMBEDDING_QUEUE, on_message_callback=callback)
    logger.info(f'[StudentEmbeddingWorker] Listening on {STUDENT_EMBEDDING_QUEUE} ...')
    channel.start_consuming()
