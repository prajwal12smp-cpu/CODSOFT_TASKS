import sys

modules = [
    'pandas',
    'numpy',
    'matplotlib',
    'seaborn',
    'sklearn',
    'joblib',
    'streamlit',
    'nltk',
]
for mod in modules:
    try:
        __import__(mod)
        print(f'{mod}: ok')
    except Exception as exc:
        print(f'{mod}: fail - {exc.__class__.__name__}: {exc}')
sys.exit(0)
