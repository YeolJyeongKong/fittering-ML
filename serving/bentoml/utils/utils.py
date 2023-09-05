import requests
import re


def local_check():
    req = requests.get("http://ipconfig.kr")
    host_ip = re.search(r"IP Address : (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", req.text)[
        1
    ]
    if host_ip == "220.72.106.46":
        return True
    else:
        return False
