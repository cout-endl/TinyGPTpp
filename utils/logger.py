import logging
import os

def get_logger(name: str = "tinygpt_plus", log_file: str = "run.log", level=logging.INFO):
    """
    获取统一格式的日志器，同时输出到控制台和文件
    重复调用不会重复添加handler
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 日志格式
    # 控制台：无时间、无模块名，纯文本对齐原版输出
    console_formatter = logging.Formatter("%(message)s")
    # 文件日志：保留完整时间戳方便排查
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 文件输出
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger
