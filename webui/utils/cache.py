import time
import uuid
import shutil
import os
import stat


def setup_workspace(folder):
    request_id = str(uuid.uuid4())
    os.makedirs(folder, exist_ok=True)

    working_dir = os.path.join(folder, request_id)
    os.makedirs(working_dir, exist_ok=True)

    log_dir = os.path.join(folder, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{request_id}.log")

    return log_file, working_dir


def on_rm_error(func, path, exc_info):
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IWRITE)

    time.sleep(0.5)
    try:
        func(path)
    except Exception:
        pass

def cleanup_workspace(working_dir):
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir, onerror=on_rm_error)
