import time

class DebugLogger:
    def __init__(self, debuglog_file_path: str):
        """初始化调试日志记录器"""
        self.debuglog_file = debuglog_file_path
        # 清空之前的debuglog
        try:
            open(self.debuglog_file, "w").close()
        except Exception as e:
            print(f"初始化debuglog文件失败: {e}")
    
    def write_debuglog(self, request_type: str, prompt: str, response: str):
        """写入debuglog"""
        try:
            with open(self.debuglog_file, "a", encoding="utf-8") as f:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                f.write(f"===== {timestamp} {request_type} =====\n")
                f.write(f"请求内容:\n{prompt}\n\n")
                f.write(f"响应内容:\n{response}\n\n")
                f.write("=" * 50 + "\n\n")
        except Exception as e:
            print(f"写入debuglog失败: {e}")