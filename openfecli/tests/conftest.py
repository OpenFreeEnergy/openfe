import urllib.error
import urllib.request

try:
    urllib.request.urlopen("https://www.google.com")
except urllib.error.URLError:  # -no-cov-
    HAS_INTERNET = False
else:
    HAS_INTERNET = True
