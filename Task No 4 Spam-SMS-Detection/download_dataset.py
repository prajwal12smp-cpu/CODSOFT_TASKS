import urllib.request

urls = [
    'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/SMSSpamCollection',
    'https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/master/data/sms.tsv',
    'https://raw.githubusercontent.com/ustcexcel/sms-spam-collection-dataset/main/spam.csv',
    'https://raw.githubusercontent.com/Chris1602/sms-spam-detection/main/spam.csv',
    'https://raw.githubusercontent.com/rishabhmisra/Spam-Classification/master/sms.csv',
    'https://raw.githubusercontent.com/muhammadfaizann/Spam-SMS-Detection/main/spam.csv',
    'https://raw.githubusercontent.com/rahulsoni1998/Spam-SMS-Classifier/main/spam.csv',
]

for url in urls:
    try:
        print(f'Trying {url}')
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=20) as response:
            data = response.read()
        print(f'SUCCESS {url} length={len(data)}')
        with open('downloaded_data.txt', 'wb') as f:
            f.write(data)
        break
    except Exception as error:
        print(f'ERROR {url}: {type(error).__name__}: {error}')
