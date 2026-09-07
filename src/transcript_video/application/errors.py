from ..process_runner import ProcessExecutionError


def describe_error(exc: Exception) -> str:
    if not isinstance(exc, ProcessExecutionError):
        return str(exc)
    stage = f"{exc.stage} failed: " if exc.stage else ""
    message = stage + str(exc)
    if exc.tool == "FFmpeg" and "Unknown encoder 'h264_nvenc'" in exc.stderr:
        message += "\nFFmpeg does not provide h264_nvenc. Try --video-encoder libx264."
    return message + "\nSee the detailed log. Run with -vv for traceback."
