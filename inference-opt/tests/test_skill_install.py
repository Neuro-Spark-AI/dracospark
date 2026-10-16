import re
import shutil
from pathlib import Path


def test_plugin_copy_retains_every_relative_skill_reference(tmp_path):
    plugin = Path(__file__).parents[1] / "plugins/inference-opt"
    installed = tmp_path / "installed-plugin"
    shutil.copytree(plugin, installed)
    for document in installed.rglob("*.md"):
        for target in re.findall(r"\]\(([^)]+)\)", document.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            file = (document.parent / target.split("#")[0]).resolve()
            assert file.is_relative_to(installed), (document, target)
            assert file.is_file(), (document, target)
