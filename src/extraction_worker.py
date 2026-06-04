import json
import logging
import pika
import requests

from config import RABBITMQ_CONN, AI_PROCESSING_QUEUE, GLINER_BASE_URL
from db import get_connection

logger = logging.getLogger(__name__)

GLINER_LABELS = ['skill', 'major', 'experience']

def build_raw_text(text_for_ai: dict) -> str:
    parts = []
    for field in ('tags', 'majors'):
        val = text_for_ai.get(field)
        if isinstance(val, list) and val:
            parts.append(' '.join(str(v) for v in val if v))
        elif isinstance(val, str) and val.strip():
            parts.append(val.strip())
    for field in ('description', 'requirements', 'experience'):
        val = text_for_ai.get(field, '')
        if val and val.strip():
            parts.append(val.strip())
    return '\n'.join(parts)


def get_feedback_dictionary() -> list:
    """
    Fetch all keywords from extraction_feedback_dictionary
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT keyword, label FROM extraction_feedback_dictionary")
        rows = cur.fetchall()
        dict_items = [{'keyword': row[0].lower(), 'label': row[1]} for row in rows]
        cur.close()
        return dict_items
    except Exception as e:
        logger.error(f"Error fetching feedback dictionary: {e}")
        return []
    finally:
        conn.close()


def extract_entities_gliner(raw_text: str) -> list:
    if not raw_text.strip():
        return []
    MAX_CHARS = 2000
    text = raw_text[:MAX_CHARS]
    entities = []
    try:
        response = requests.post(
            f"{GLINER_BASE_URL}/predict",
            json={"text": text, "labels": GLINER_LABELS, "threshold": 0.5},
            timeout=10
        )
        if response.status_code == 200:
            entities = response.json().get("entities", [])
    except Exception as e:
        logger.error(f"[ExtractionWorker] Error calling GLiNER API: {e}")
    return entities

def extract_entities_dictionary(raw_text: str, dict_items: list) -> list:
    raw_text_lower = raw_text.lower()
    entities = []
    for item in dict_items:
        kw = item['keyword']
        if kw and kw in raw_text_lower:
            entities.append({"text": kw, "label": item['label']})
    return entities


def process_message(payload: dict):
    job_id = payload.get('internal_job_id')
    text_for_ai = payload.get('text_for_ai', {})

    if not job_id:
        logger.warning('[ExtractionWorker] Missing internal_job_id')
        return

    raw_text = build_raw_text(text_for_ai)

    gliner_entities = extract_entities_gliner(raw_text)
    dict_items = get_feedback_dictionary()
    dict_entities = extract_entities_dictionary(raw_text, dict_items)

    seen_texts = set()
    final_entities = []
    
    for ent in dict_entities:
        t = ent['text'].lower()
        if t not in seen_texts:
            seen_texts.add(t)
            final_entities.append(ent)
            
    for ent in gliner_entities:
        t = ent['text'].lower()
        if t not in seen_texts:
            seen_texts.add(t)
            final_entities.append(ent)

    conn = get_connection()
    try:
        cur = conn.cursor()
        draft_json = json.dumps(final_entities, ensure_ascii=False)
        cur.execute(
            """
            UPDATE job_postings
               SET draft_extracted_metadata = %s::jsonb,
                   status = 'PENDING_REVIEW'
             WHERE id = %s
            """,
            (draft_json, job_id),
        )
        rc = cur.rowcount
        conn.commit()
        cur.close()
        logger.info(f"[ExtractionWorker] Job {job_id} drafted with {len(final_entities)} entities.")
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def run():
    params  = pika.URLParameters(RABBITMQ_CONN)
    params.heartbeat = 0
    conn    = pika.BlockingConnection(params)
    channel = conn.channel()
    channel.queue_declare(queue=AI_PROCESSING_QUEUE, durable=True)
    channel.basic_qos(prefetch_count=1)

    def callback(ch, method, properties, body):
        try:
            payload = json.loads(body)
            process_message(payload)
            if ch.is_open:
                ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            logger.error(f'[ExtractionWorker] Error: {e}', exc_info=True)
            if ch.is_open:
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    channel.basic_consume(queue=AI_PROCESSING_QUEUE, on_message_callback=callback)
    logger.info(f'[ExtractionWorker] Listening on {AI_PROCESSING_QUEUE} ...')
    channel.start_consuming()

if __name__ == "__main__":
    run()
