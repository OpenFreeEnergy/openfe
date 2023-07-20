"""
sphinxcontrib-sass
https://github.com/attakei-lab/sphinxcontrib-sass
Kayuza Takei
Apache 2.0

Modified to:
- Write directly to Sphinx output directory
- Infer targets if not given
- Ensure ``target: Path`` in ``configure_path()``
- Return version number and thread safety from ``setup()``
- Use compressed style by default
- More complete type checking
"""

from os import PathLike
from pathlib import Path
from typing import Optional, Union


import sass
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.util import logging


logger = logging.getLogger(__name__)


def configure_path(conf_dir: str, src: Optional[Union[PathLike, Path]]) -> Path:
    if src is None:
        target = Path(conf_dir)
    else:
        target = Path(src)
    if not target.is_absolute():
        target = Path(conf_dir) / target
    return target


def get_targets(app: Sphinx) -> dict[Path, Path]:
    src_dir = configure_path(app.confdir, app.config.sass_src_dir)
    dst_dir = configure_path(app.outdir, app.config.sass_out_dir)

    if isinstance(app.config.sass_targets, dict):
        targets = app.config.sass_targets
    else:
        targets = {
            path: path.relative_to(src_dir).with_suffix(".css")
            for path in src_dir.glob("**/[!_]*.s[ca]ss")
        }

    return {src_dir / src: dst_dir / dst for src, dst in targets.items()}


def build_sass_sources(app: Sphinx, env: BuildEnvironment):
    logger.debug("Building stylesheet files")
    include_paths = [str(p) for p in app.config.sass_include_paths]
    targets = get_targets(app)
    output_style = app.config.sass_output_style
    # Build css files
    for src, dst in targets.items():
        content = src.read_text()
        css = sass.compile(
            string=content,
            output_style=output_style,
            include_paths=[str(src.parent)] + include_paths,
        )
        dst.parent.mkdir(exist_ok=True, parents=True)
        dst.write_text(css)


def setup(app: Sphinx):
    """
    Setup function for this extension.
    """
    logger.debug(f"Using {__name__}")
    app.add_config_value("sass_include_paths", [], "html")
    app.add_config_value("sass_src_dir", None, "html")
    app.add_config_value("sass_out_dir", None, "html")
    app.add_config_value("sass_targets", None, "html")
    app.add_config_value("sass_output_style", "compressed", "html")
    app.connect("env-updated", build_sass_sources)

    return {
        "version": "0.3.4ofe",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
