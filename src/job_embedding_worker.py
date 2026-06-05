import json
import logging
import pika

from config import RABBITMQ_CONN, JOB_EMBEDDING_QUEUE
from db import get_connection

logger = logging.getLogger(__name__)

def generate_embedding(bge_model, clean_text: str) -> list | None:
    if not clean_text or not clean_text.strip():
        return None
    output = bge_model.encode([clean_text], batch_size=1, max_length=512)
    vec = output['dense_vecs'][0]
    return vec.tolist()

def save_embeddings_to_db(job_id: str, prof_embedding: list, major_embedding: list | None):
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
        rc = cur.rowcount
        conn.commit()
        cur.close()
        if rc == 0:
            logger.warning(f"0 rows updated for job_id {job_id}")
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def process_message(bge_model, payload: dict):
    job_id = payload.get('job_id')
    final_labels = payload.get('final_labels', [])

    if not job_id:
        logger.warning('[JobEmbeddingWorker] Missing job_id')
        return

    # Extract text by label
    skills_exp = []
    majors = []
    for item in final_labels:
        label = item.get('label', '').lower()
        text = item.get('text', '')
        if label in ['skill', 'experience']:
            skills_exp.append(text)
        elif label == 'major':
            majors.append(text)

    skill_exp_text = ', '.join(skills_exp)
    major_text = ', '.join(majors)

    if not skill_exp_text:
        logger.warning(f'[JobEmbeddingWorker] No skill/experience for job {job_id}')
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute("UPDATE job_postings SET status = 'DONE' WHERE id = %s", (job_id,))
            conn.commit()
            cur.close()
        except Exception as e:
            conn.rollback()
        finally:
            conn.close()
        return

    prof_embedding = generate_embedding(bge_model, skill_exp_text)
    major_embedding = generate_embedding(bge_model, major_text) if major_text else None

    if prof_embedding is None:
        logger.warning(f'[JobEmbeddingWorker] Empty professional text, skipping job {job_id}')
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute("UPDATE job_postings SET status = 'DONE' WHERE id = %s", (job_id,))
            conn.commit()
            cur.close()
        except Exception as e:
            conn.rollback()
        finally:
            conn.close()
        return

    save_embeddings_to_db(job_id, prof_embedding, major_embedding)
    logger.info(f'[JobEmbeddingWorker] OK Job {job_id} -> DONE')


def run(bge_model):
    params  = pika.URLParameters(RABBITMQ_CONN)
    params.heartbeat = 0
    conn    = pika.BlockingConnection(params)
    channel = conn.channel()
    channel.queue_declare(queue=JOB_EMBEDDING_QUEUE, durable=True)
    channel.basic_qos(prefetch_count=1)

    def callback(ch, method, properties, body):
        try:
            payload = json.loads(body)
            process_message(bge_model, payload)
            if ch.is_open:
                ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            logger.error(f'[JobEmbeddingWorker] Error: {e}', exc_info=True)
            if ch.is_open:
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    channel.basic_consume(queue=JOB_EMBEDDING_QUEUE, on_message_callback=callback)
    logger.info(f'[JobEmbeddingWorker] Listening on {JOB_EMBEDDING_QUEUE} ...')
    channel.start_consuming()
