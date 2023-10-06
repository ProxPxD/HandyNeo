import pathlib
import re

from setuptools import setup, find_packages

main_dir = pathlib.Path(__file__).parent.resolve()
readme_path = main_dir / "README.md"
LONG_DESCRIPTION = readme_path.read_text() if readme_path.exists() else 'LONG DESCRIPTION'


VERSION = '0.0.9'
DESCRIPTION = 'Facade for neo4j to create nodes and relationships in a more robust way'

# Setting up
# exit()
PACKAGE_NAME = 'handyneo'
SOURCE_DIRECTORY = 'src'

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author="ProxPxD (Piotr Maliszewski)",
    # author_email="<mail@neuralnine.com>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=[PACKAGE_NAME],
    package_dir={PACKAGE_NAME: SOURCE_DIRECTORY},
    keywords=['python', 'handy', 'handyneo', 'neo', 'neo4j'],
    classifiers=[],
    include_package_data=True,
)