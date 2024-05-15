from datetime import datetime

class Logger:
    def __init__(self, tag, enabled):
        self.tag = tag
        self.enabled = enabled

    def log(self,  log_str,  ):
        date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        if (self.enabled):
            print ( f"[{date_time}] [{self.tag}] {log_str}"  )