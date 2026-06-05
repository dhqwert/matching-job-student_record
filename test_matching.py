import os
import sys
import traceback
from src.matching_worker import run_matching
os.environ['DATABASE_URL'] = 'postgresql://postgres:postgres@localhost:5432/student_360'

try:
    print("Running matching for BCS230070...")
    run_matching({
        "student_id": "3e415d00-a8fa-49d1-846d-215c3f6923dc",
        "filters": {}
    })
    print("Done")
except Exception as e:
    traceback.print_exc()
