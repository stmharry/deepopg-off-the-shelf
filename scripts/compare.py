import fnmatch
import os
import re
from pathlib import Path

from absl import app, flags, logging

flags.DEFINE_string("root_dir", None, "Root directory")
flags.DEFINE_integer("height", 400, "Height of images")
flags.DEFINE_list("image_patterns", None, "Image patterns")
flags.DEFINE_string("output_html_path", None, "Output HTML file")
FLAGS = flags.FLAGS


def relative_to(path: Path, start: Path) -> Path:
    return Path(os.path.relpath(path, start=start))


def td(path: Path, height: int) -> str:
    return f"<td><a href='{path}' target='_blank'><img src='{path}' height='{height}px'></a></td>"


def tr(name: str, paths: list[Path], height: int) -> str:
    s: str = ""

    s += f"<tr>\n"
    s += f"<th scope='row'>{name}</td>\n"

    for path in paths:
        s += f"{td(path, height=height)}\n"

    s += f"</tr>\n"

    return s


def path_by_stem(pattern: Path) -> dict[str, Path]:
    current_dir: Path = Path.cwd()
    relative_pattern: Path = relative_to(pattern, current_dir)
    full_pattern: Path = Path(current_dir, relative_pattern)

    full_pattern_regex: str = fnmatch.translate(str(full_pattern)).replace(
        ".*", "(.*)"
    )  # to capture

    path_by_stem: dict[str, Path] = {}
    for path in Path(current_dir).glob(str(relative_pattern)):
        match = re.fullmatch(full_pattern_regex, str(path))
        if match is None:
            raise ValueError(f"Unexpected path: {path}")

        path_by_stem[match.group(1)] = path

    return path_by_stem


def main(_):
    logging.set_verbosity(logging.INFO)

    stem_to_path_list: list[dict[str, Path]] = []
    for pattern in FLAGS.image_patterns:
        full_pattern: Path = Path(FLAGS.root_dir, pattern)
        stem_to_path_list.append(path_by_stem(full_pattern))

    stems: set[str] = set(stem_to_path_list[0]).intersection(*stem_to_path_list[1:])

    output_html_path: Path = Path(FLAGS.root_dir, FLAGS.output_html_path)
    output_dir: Path = output_html_path.parent

    #

    html_head: str = """
    <style>
    th[scope=col] {
      background-color: white;
      position: -webkit-sticky;
      position: sticky;
      top: 0;
      z-index: 2;
    }
    th[scope=row] {
      background-color: white;
      position: -webkit-sticky;
      position: sticky;
      left: 0;
      z-index: 1;
    }
    </style>
    """

    html_body: str = "\n".join(
        [
            f"<table>",
            f"  <thead>",
            f"    <th scope='col'>Name</th>",
            *[
                f"    <th scope='col'>{pattern}</th>"
                for pattern in FLAGS.image_patterns
            ],
            f"  </thead>",
            f"  <tbody>",
            *[
                tr(
                    name=stem,
                    paths=[
                        relative_to(stem_to_path[stem], start=output_dir)
                        for stem_to_path in stem_to_path_list
                    ],
                    height=FLAGS.height,
                )
                for stem in sorted(stems)
            ],
            f"  </tbody>",
            f"</table>",
        ]
    )

    html: str = f"""
    <!DOCTYPE html>
    <html>
    <head>
      {html_head}
    </head>
    <body>
      {html_body}
    </body>
    </html>
    """

    logging.info(f"Writing to {output_html_path}")

    with open(output_html_path, "w") as f:
        f.write(html)


if __name__ == "__main__":
    app.run(main)
