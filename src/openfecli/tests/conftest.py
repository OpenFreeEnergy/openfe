import urllib.error
import urllib.request

import pooch

try:
    urllib.request.urlopen("https://www.google.com")
except urllib.error.URLError:  # -no-cov-
    HAS_INTERNET = False
else:
    HAS_INTERNET = True

POOCH_CACHE = pooch.os_cache("openfe")
