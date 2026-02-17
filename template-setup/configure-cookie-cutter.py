# noqa: D100
import argparse
import json
from collections import OrderedDict
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument("repository", type=str)
parser.add_argument("author", type=str)
parser.add_argument("output", type=str)

args = parser.parse_args()

output_file = Path(args.output)

template = OrderedDict(
    repository=args.repository,
    author=args.author,
    _copy_without_render=[".github", ".git"],
    org="AZ-AI",
    package_name="{{cookiecutter.repository.lower().replace(' ', '_').replace('-', '_')}}",
)


config = json.dumps(template, indent=4)
output_file.write_text(config, encoding="utf-8")
