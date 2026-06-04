"""
matching_worker.py — Consumer: match_request_queue

Flow:
  1. Nhận request từ match_request_queue
  2. Fetch student professional_embedding + structural preferences từ DB
  3. Fetch candidate job_postings có professional_embedding (status='DONE')
     - Optional hard SQL filters: gender, salary
  4. Cosine similarity giữa student vector và mỗi job vector
  5. INSERT/UPDATE match_results table

NOTE về Location:
  - Location được trả kèm trong match_details (để hiển thị kết quả) 
  - Nhưng KHÔNG đưa vào hàm tính cosine similarity.

Payload expected (từ backend):
{
  "student_id": "uuid",
  "filters": {                     // optional hard filters
    "gender":     "MALE",          // lọc job yêu cầu gender này hoặc ANY
    "salary_min": 10000000         // lọc job có salary_max >= giá trị này
  }
}

match_results schema (từ entity):
  student_id, job_id, match_percent (int 0-100),
  model_version, processing_time_ms, match_details (jsonb)
"""
import json
import time
import logging
import numpy as np
import pika
import psycopg2.extras
import redis

from config import RABBITMQ_CONN, MATCH_REQUEST_QUEUE, REDIS_CONN
from db import get_connection

logger = logging.getLogger(__name__)

# Initialize Redis client
redis_client = redis.from_url(REDIS_CONN)

MODEL_VERSION = 'v2-bge-m3'


# ── Vector math ────────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1]. Returns 0 if either vector is zero."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def similarity_to_percent(sim: float) -> int:
    """Map cosine similarity [-1,1] → percent [0,100]."""
    return max(0, min(100, int((sim + 1) / 2 * 100)))


# ── DB helpers ─────────────────────────────────────────────────────────────────

def fetch_student(conn, student_id: str) -> dict | None:
    """
    Lấy professional_embedding và thông tin cấu trúc của student.
    Trả về None nếu không tìm thấy hoặc chưa có embedding.
    """
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute(
        """
        SELECT sr.id,
               sr.professional_embedding::text AS embedding_str,
               sr.major_embedding::text AS major_embedding_str,
               sr.gender,
               sr.salary_min,
               sr.salary_max,
               sr.location,
               sr.career_profile
          FROM student_records sr
         WHERE sr.id = %s
        """,
        (student_id,),
    )
    row = cur.fetchone()
    cur.close()
    if row is None or row['embedding_str'] is None:
        return None
    return dict(row)


def fetch_candidate_jobs(conn, filters: dict) -> list[dict]:
    """
    Fetch jobs có professional_embedding.
    Áp dụng hard SQL filter nếu được cung cấp trong filters.
    """
    conditions = ["status = 'DONE'", "professional_embedding IS NOT NULL"]
    params = []

    gender_filter = filters.get('gender')
    if gender_filter and gender_filter in ('MALE', 'FEMALE'):
        conditions.append(
            "(basic_info->>'gender' = 'ANY' OR basic_info->>'gender' = %s)"
        )
        params.append(gender_filter)

    salary_min = filters.get('salary_min')
    if salary_min and salary_min > 0:
        conditions.append(
            "(working_conditions->>'salary_min')::numeric >= %s"
        )
        params.append(salary_min)

    salary_max = filters.get('salary_max')
    if salary_max and salary_max > 0:
        conditions.append(
            "(working_conditions->>'salary_max')::numeric <= %s"
        )
        params.append(salary_max)

    currency = filters.get('currency')
    if currency:
        conditions.append(
            "working_conditions->>'currency' = %s"
        )
        params.append(currency)

    is_negotiable = filters.get('is_negotiable')
    if is_negotiable is not None:
        val = 'true' if is_negotiable else 'false'
        conditions.append(
            "working_conditions->>'is_negotiable' = %s"
        )
        params.append(val)

    keyword = filters.get('keyword')
    if keyword:
        conditions.append(
            "(job_title ILIKE %s OR display_content::text ILIKE %s)"
        )
        like_str = f"%{keyword}%"
        params.extend([like_str, like_str])

    where = ' AND '.join(conditions)
    query = f"""
        SELECT id,
               job_title,
               professional_embedding::text AS embedding_str,
               major_embedding::text AS major_embedding_str,
               basic_info->>'locations'     AS locations_raw,
               basic_info->>'gender'        AS gender,
               working_conditions->>'salary_min'    AS salary_min,
               working_conditions->>'salary_max'    AS salary_max,
               working_conditions->>'is_negotiable' AS is_negotiable,
               working_conditions->>'currency'      AS currency,
               source_metadata->>'original_url'     AS source_url
          FROM job_postings
         WHERE {where}
    """
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute(query, params)
    rows = [dict(r) for r in cur.fetchall()]
    cur.close()
    return rows


def parse_vector(embedding_str: str) -> np.ndarray | None:
    """Convert pgvector string '[0.1,0.2,...]' to numpy array."""
    if not embedding_str:
        return None
    try:
        cleaned = embedding_str.strip().strip('[]')
        return np.array([float(x) for x in cleaned.split(',')], dtype=np.float32)
    except Exception:
        return None


def upsert_match_result(conn, student_id: str, job_id: str,
                         match_percent: int, processing_time_ms: int,
                         match_details: dict):
    """INSERT ON CONFLICT UPDATE match_results."""
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO match_results
               (student_id, job_id, match_percent, model_version,
                processing_time_ms, match_details)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (student_id, job_id)
        DO UPDATE SET
            match_percent      = EXCLUDED.match_percent,
            model_version      = EXCLUDED.model_version,
            processing_time_ms = EXCLUDED.processing_time_ms,
            match_details      = EXCLUDED.match_details,
            updated_at         = NOW()
        """,
        (
            student_id, job_id, match_percent, MODEL_VERSION,
            processing_time_ms, json.dumps(match_details),
        ),
    )
    cur.close()


# ── Core matching logic ─────────────────────────────────────────────────────────

def run_matching(payload: dict):
    student_id     = payload.get('student_id')
    filters        = payload.get('filters', {})

    if not student_id:
        logger.warning('[MatchingWorker] Missing student_id — skipping')
        return

    t_start = time.time()
    conn    = get_connection()

    try:
        # 1. Fetch student
        student = fetch_student(conn, student_id)
        if student is None:
            logger.warning(f'[MatchingWorker] Student {student_id} not found or has no embedding')
            return

        student_vec = parse_vector(student['embedding_str'])
        if student_vec is None:
            logger.warning(f'[MatchingWorker] Invalid embedding for student {student_id}')
            return

        student_major_vec = parse_vector(student.get('major_embedding_str'))

        # 2. Fetch candidate jobs (with optional hard filters)
        jobs = fetch_candidate_jobs(conn, filters)
        logger.info(f'[MatchingWorker] Student {student_id}: {len(jobs)} candidate jobs after filtering')

        if not jobs:
            logger.info(f'[MatchingWorker] No candidates found for student {student_id}')
            return

        # 3. Compute cosine similarity for each job
        results = []
        for job in jobs:
            job_vec = parse_vector(job['embedding_str'])
            if job_vec is None:
                continue
                
            job_major_vec = parse_vector(job.get('major_embedding_str'))
            
            prof_sim = cosine_similarity(student_vec, job_vec)

            if student_major_vec is not None and job_major_vec is not None:
                # Cả hai đều có major embedding → tính cosine thực
                major_sim = cosine_similarity(student_major_vec, job_major_vec)
            elif student_major_vec is not None and job_major_vec is None:
                # Student có major rõ ràng nhưng job không có → không rõ major
                # Dùng giá trị penalty thấp để kích hoạt giảm điểm
                major_sim = 0.3
            else:
                # Student chưa có major embedding → bỏ qua bộ lọc major
                major_sim = 1.0
                
            if major_sim < 0.50:
                prof_sim *= 0.5
                
            percent = similarity_to_percent(prof_sim)
            results.append((job, percent, prof_sim, major_sim))

        # Sort descending by similarity
        results.sort(key=lambda x: x[2], reverse=True)

        # 4. Save results (to Redis if realtime filter, else to DB)
        elapsed_ms = int((time.time() - t_start) * 1000)
        saved = 0
        
        is_realtime = False
        if filters and any(v is not None and v != '' and v != 0 for v in filters.values()):
            is_realtime = True

        if is_realtime:
            # Save to Redis for 1 hour
            redis_data = []
            for job, percent, sim, major_sim in results:
                redis_data.append({
                    "job_id": job['id'],
                    "match_percent": percent,
                    "model_version": MODEL_VERSION,
                    "match_details": {
                        'cosine_score':  round(sim, 4),
                        'major_sim':     round(major_sim, 4),
                        'match_percent': percent,
                        'locations':     job.get('locations_raw', ''),
                        'salary_min':    job.get('salary_min'),
                        'salary_max':    job.get('salary_max'),
                        'currency':      job.get('currency'),
                        'is_negotiable': job.get('is_negotiable'),
                        'source_url':    job.get('source_url'),
                    }
                })
            redis_client.set(f"realtime_match:{student_id}", json.dumps(redis_data), ex=3600)
            saved = len(results)
            logger.info(f'[MatchingWorker] Saved {saved} results to Redis (realtime_match:{student_id})')
        else:
            # Save to match_results DB
            for job, percent, sim, major_sim in results:
                match_details = {
                    'cosine_score':  round(sim, 4),
                    'major_sim':     round(major_sim, 4),
                    'match_percent': percent,
                    # Location attached for display — NOT used in similarity calc
                    'locations':     job.get('locations_raw', ''),
                    'salary_min':    job.get('salary_min'),
                    'salary_max':    job.get('salary_max'),
                    'currency':      job.get('currency'),
                    'is_negotiable': job.get('is_negotiable'),
                    'source_url':    job.get('source_url'),
                }
                try:
                    upsert_match_result(
                        conn, student_id, job['id'],
                        percent, elapsed_ms, match_details,
                    )
                    saved += 1
                except Exception as e:
                    logger.error(f'[MatchingWorker] Upsert failed for job {job["id"]}: {e}')
                    conn.rollback()

            conn.commit()

        total_ms = int((time.time() - t_start) * 1000)
        logger.info(
            f'[MatchingWorker] ✓ Student {student_id}: {saved}/{len(results)} results processed '
            f'| top_score={results[0][1] if results else 0}% | {total_ms}ms'
        )

    except Exception as e:
        conn.rollback()
        logger.error(f'[MatchingWorker] Fatal error for student {student_id}: {e}', exc_info=True)
    finally:
        conn.close()


# ── RabbitMQ consumer ───────────────────────────────────────────────────────────

def run():
    params  = pika.URLParameters(RABBITMQ_CONN)
    params.heartbeat = 0
    conn    = pika.BlockingConnection(params)
    channel = conn.channel()
    channel.queue_declare(queue=MATCH_REQUEST_QUEUE, durable=True)
    channel.basic_qos(prefetch_count=1)

    def callback(ch, method, properties, body):
        try:
            payload = json.loads(body)
            logger.info(f'[MatchingWorker] Received match request: student_id={payload.get("student_id")}')
            run_matching(payload)
            if ch.is_open:
                ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            logger.error(f'[MatchingWorker] Error processing message: {e}', exc_info=True)
            if ch.is_open:
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    channel.basic_consume(queue=MATCH_REQUEST_QUEUE, on_message_callback=callback)
    logger.info(f'[MatchingWorker] Listening on {MATCH_REQUEST_QUEUE} ...')
    channel.start_consuming()
