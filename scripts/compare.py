import fnmatch
import os
import re
from pathlib import Path

import pandas as pd
from absl import app, flags, logging

flags.DEFINE_list(
    "image_patterns",
    [],
    "Image path patterns to search for. Use '*' for wildcard.",
)
flags.DEFINE_string("output_html_path", "./index.html", "Output HTML path.")
flags.DEFINE_integer("height", 400, "Height of images in the HTML.")
FLAGS = flags.FLAGS


def relative_to(path: Path, start: Path) -> Path:
    return Path(os.path.relpath(path, start=start))


def glob(pattern: Path, root_dir: Path | None = None) -> pd.Series:
    dummy_dir: Path = Path("/")

    relative_pattern: Path = relative_to(pattern, dummy_dir)
    full_pattern: Path = Path(dummy_dir, relative_pattern)
    full_pattern_regex: str = fnmatch.translate(str(full_pattern)).replace(
        ".*", "(.*)"
    )  # to capture group
    full_pattern_obj: re.Pattern = re.compile(full_pattern_regex)

    id_to_path: dict[str, Path] = {}
    for path in Path(dummy_dir).glob(str(relative_pattern)):
        match = full_pattern_obj.match(str(path))
        if match is None:
            raise ValueError(f"Unexpected path: {path}")

        if root_dir is not None:
            path = relative_to(path, start=root_dir)

        id_to_path["-".join(match.groups())] = path

    return pd.Series(id_to_path, name=str(pattern))


def th(text: str, scope: str) -> str:
    return f"<th scope='{scope}'>{text}</th>"


def td(path: Path, height: int) -> str:
    return f"""
    <td>
        <a href='{path}' target='_blank'>
            <img src='{path}' height='{height}px' />
        </a>
    </td>
    """


def tr(header: str, paths: list[Path]) -> str:
    return f"""
    <tr>
        {th(header, scope="row")}
        {"".join(td(path, height=FLAGS.height) for path in paths)}
    </tr>
    """


def table(df: pd.DataFrame, require_full_row: bool = True) -> str:
    trs: list[str] = [
        tr(header=str(id), paths=row.tolist())
        for id, row in df.iterrows()
        if require_full_row and row.notna().all()
    ]

    return f"""
    <table>
        <thead>
            <th scope='col'>Name</th>
            {"".join(f"<th scope='col'>{column}</th>" for column in df.columns)}
        </thead>
        <tbody>
            {"".join(trs)}
        </tbody>
    </table>
    """


def html(df: pd.DataFrame) -> str:
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            table, th, td {{
                border: 1px solid black;
                border-collapse: separate;
            }}
            th[scope=col] {{
              background-color: white;
              position: -webkit-sticky;
              position: sticky;
              top: 0;
              z-index: 2;
            }}
            th[scope=row] {{
              background-color: white;
              position: -webkit-sticky;
              position: sticky;
              left: 0;
              z-index: 1;
            }}
        </style>
    </head>
    <body>
        {table(df=df)}
    </body>
    </html>
    """


def main(_):
    output_html_path: Path = Path(FLAGS.output_html_path)
    output_dir: Path = output_html_path.parent

    df: pd.DataFrame = pd.concat(
        [glob(Path(pattern), root_dir=output_dir) for pattern in FLAGS.image_patterns],
        axis=1,
    ).sort_index()
    html_str: str = html(df=df)

    logging.info(f"Writing to {output_html_path}")
    with open(output_html_path, "w") as f:
        f.write(html_str)


if __name__ == "__main__":
    app.run(main)
