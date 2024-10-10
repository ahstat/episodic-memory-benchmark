def no_ssl_verification():
    import requests.adapters
    import urllib3

    # Ignore SSL warnings
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    original_send = requests.adapters.HTTPAdapter.send

    def send(*args, **kwargs):
        kwargs['verify'] = False
        return original_send(*args, **kwargs)
    
    requests.adapters.HTTPAdapter.send = send
