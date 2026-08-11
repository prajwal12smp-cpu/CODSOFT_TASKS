modules = ['sklearn', 'joblib', 'nltk', 'numpy']
for mod in modules:
    try:
        module = __import__(mod)
        print(f'{mod}: ok')
    except Exception as exc:
        print(f'{mod}: fail - {exc.__class__.__name__}: {exc}')
