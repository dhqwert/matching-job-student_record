import os
from src.matching_worker import run_matching
os.environ['DATABASE_URL'] = 'postgresql://postgres:postgres@localhost:5432/student_360'

run_matching({
    "student_id": "00038545-6307-4e5f-a585-537dbb6ee065",
    "filters": {}
})
