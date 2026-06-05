"""
config.py — Load environment variables for ai_service.
"""
import os
from dotenv import load_dotenv

# Disable symlinks on Windows to fix "WinError 1314 A required privilege is not held"
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'

load_dotenv()

# ── Database ─────────────────────────────────────────────────────────────────
DB_HOST     = os.getenv('DB_HOST',     'localhost')
DB_PORT     = int(os.getenv('DB_PORT', '5433'))
DB_USER     = os.getenv('DB_USER',     'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'postgres')
DB_DATABASE = os.getenv('DB_DATABASE', 'iam')

# ── RabbitMQ ─────────────────────────────────────────────────────────────────
RABBITMQ_CONN           = os.getenv('RABBITMQ_CONN', 'amqp://agi_rabbitmq_user:agi_rabbitmq_user@localhost:5672/agi_rabbitmq_user')
AI_PROCESSING_QUEUE     = os.getenv('AI_PROCESSING_QUEUE', 'ai_processing_queue')
MATCH_REQUEST_QUEUE     = os.getenv('MATCH_REQUEST_QUEUE', 'match_request_queue')
STUDENT_EMBEDDING_QUEUE = os.getenv('STUDENT_EMBEDDING_QUEUE', 'student_embedding_queue')
JOB_EMBEDDING_QUEUE     = os.getenv('JOB_EMBEDDING_QUEUE', 'job_embedding_queue')
# ── Matching ──────────────────────────────────────────────────────────────────
RERANKING_LIMIT         = int(os.getenv('RERANKING_LIMIT', '200'))

# ── External APIs ────────────────────────────────────────────────────────────
GLINER_BASE_URL = os.getenv('GLINER_BASE_URL', 'http://localhost:7777')
BACKEND_API_URL = os.getenv('BACKEND_API_URL', 'http://localhost:3457/api/v1')
AI_API_KEY      = os.getenv('AI_API_KEY', '')

# ── Redis ────────────────────────────────────────────────────────────────────
REDIS_CONN = os.getenv('REDIS_CONN', 'redis://localhost:6379/0')

# ── Model config ──────────────────────────────────────────────────────────────
# GLiNER base model — downloaded once, cached locally
GLINER_MODEL_NAME = os.getenv('GLINER_MODEL_NAME', 'urchade/gliner_base')

# BGE-M3 embedding model — downloaded once, cached locally
BGE_MODEL_NAME    = os.getenv('BGE_MODEL_NAME', 'BAAI/bge-m3')

MODEL_VERSION     = os.getenv('MODEL_VERSION', 'JobCare-v1')

# Local cache dir — models live here permanently, no re-download on restart
MODEL_CACHE_DIR   = os.path.abspath(os.getenv('MODEL_CACHE_DIR', './model_cache'))

# HuggingFace token (optional, for faster downloads)
HF_TOKEN = os.getenv('HF_TOKEN', '')
if HF_TOKEN:
    os.environ['HF_TOKEN'] = HF_TOKEN
    os.environ['HUGGINGFACE_HUB_TOKEN'] = HF_TOKEN
