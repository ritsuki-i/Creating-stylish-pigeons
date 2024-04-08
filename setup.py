from setuptools import setup, find_packages

setup(
    name='stylechange',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,  # これにより、MANIFEST.inで指定されたファイルが含まれます
    package_data={
        # 'パッケージ名': ['ディレクトリ/ファイル']
        'stylechange': ['img_e_g/*'],
    },
)