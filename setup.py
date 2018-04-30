from setuptools import setup
from collections import defaultdict
from pip.req import parse_requirements

requirements = []
extras = defaultdict(list)
for r in parse_requirements('requirements.txt', session='hack'):
    if r.markers:
        extras[':' + str(r.markers)].append(str(r.req))
    else:
        requirements.append(str(r.req))

setup(name='TFServingSpark',
      version='0.1.0',
      packages=['tss'],
          install_requires=requirements,
    extras_require=extras,
      description='Tensorflow serving on spark dataframe',
      long_description="""
        tss is a machine learning toolkit designed to
        apply tensorflow saved model to spark dataframe.
        """,
        author='Peng Yu',
        author_email='yupbank@gmail.com',
        url='https://github.com/yupbank/tf-spark-serving',
        download_url='https://github.com/yupbank/tf-spark-serving/archive/0.1.0.tar.gz',
        keywords=['tensorflow', 'machine learning', 'dataframe', 'spark', 'model-serving'],
        classifiers=[],
)
