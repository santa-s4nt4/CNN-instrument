from google_images_download import google_images_download

response = google_images_download.googleimagesdownload()

arguments = {"keywords": "piano", "limit": 10000,
             "chromedriver": "C:\\Users\\santa\\chromedriver_win32\\chromedriver.exe", "print_urls": True}
response.download(arguments)
