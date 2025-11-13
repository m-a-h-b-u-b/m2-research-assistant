"""
M2 Research Assistant
Author  : Md Mahbubur Rahman
License : Apache 2.0  
GitHub  : https://github.com/m-a-h-b-u-b/m2-research-assistant
URL     : https://m-a-h-b-u-b.github.io 
"""
import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_environment(tmp_path_factory):
os.environ["CHROMA_DB_DIR"] = str(tmp_path_factory.mktemp("chroma"))