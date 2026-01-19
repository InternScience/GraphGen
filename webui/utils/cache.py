import os
import shutil
import stat
import time
import uuid


def setup_workspace(folder):
    request_id = str(uuid.uuid4())
    os.makedirs(folder, exist_ok=True)

    working_dir = os.path.join(folder, request_id)
    os.makedirs(working_dir, exist_ok=True)

    log_dir = os.path.join(folder, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{request_id}.log")

    return log_file, working_dir


def cleanup_workspace(working_dir):
    if not os.path.exists(working_dir):
        return
    st = os.stat(working_dir)
    os.chmod(working_dir, st.st_mode | stat.S_IWRITE)

    time.sleep(0.5)
    try:
        shutil.rmtree(working_dir)
    except Exception:
        pass
