import os
import sys
import time
from src.matching_worker import fetch_student, fetch_candidate_jobs, parse_vector, cosine_similarity
import psycopg2
import psycopg2.extras

from db import get_connection

student_id = "3e415d00-a8fa-49d1-846d-215c3f6923dc"
conn = get_connection()

t0 = time.time()
student = fetch_student(conn, student_id)
t1 = time.time()
print(f"fetch_student took {t1 - t0:.3f}s")

student_vec = parse_vector(student['embedding_str'])
student_major_vec_str = student.get('major_embedding_str')
t2 = time.time()
print(f"parse_vector student took {t2 - t1:.3f}s")

jobs = fetch_candidate_jobs(conn, {}, student_major_vec_str, limit=200)
t3 = time.time()
print(f"fetch_candidate_jobs took {t3 - t2:.3f}s")

for job in jobs:
    job_vec = parse_vector(job['embedding_str'])
    prof_sim = cosine_similarity(student_vec, job_vec)
t4 = time.time()
print(f"cosine similarities for {len(jobs)} jobs took {t4 - t3:.3f}s")
