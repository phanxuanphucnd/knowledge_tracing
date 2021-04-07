import os
from setuptools import setup, find_packages

try:
    # pip >=20
    from pip._internal.network.session import PipSession
    from pip._internal.req import parse_requirements
except ImportError:
    try:
        # 10.0.0 <= pip <= 19.3.1
        from pip._internal.download import PipSession
        from pip._internal.req import parse_requirements
    except ImportError:
        # pip <= 9.0.3
        from pip.download import PipSession
        from pip.req import parse_requirements

long_description = ''

# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements('requirements.txt', session=False)

# reqs is a list of requirement
# e.g. ['django==1.5.1', 'mezzanine==1.4.6']
try:
    reqs = [str(ir.req) for ir in install_reqs]
except:
    reqs = [str(ir.requirement) for ir in install_reqs]

VERSION = os.getenv('PACKAGE_VERSION', 'v0.0.1')[1:]

setup(
    name='anivia',
    version=VERSION,
    description='anivia is a library for Knowledge Tracing module in Adaptive learning system.',
    long_description=long_description,
    url='https://github.com/phanxuanphucnd/knowledge_tracing',
    packages=find_packages(),
    include_package_data=True,
    package_data={'anivia': ["*.*"], },
    author='phucpx',
    author_email='phanxuanphucnd@gmail.com',
    classifiers=[
        'Programming Language :: Python :: 3.x',
    ],
    install_requires=reqs,
    keywords='anivia',
    python_requires='>=3.6',
    py_modules=['anivia'],
    # entry_points={
    #     'console_scripts': [
    #         'denver = denver.run_cli:entry_point'
    #     ]
    # },

)
