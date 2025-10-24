import urllib.request

try:
    urllib.request.urlopen("https://www.google.com")
except:  # -no-cov-
    HAS_INTERNET = False
else:
    HAS_INTERNET = True
