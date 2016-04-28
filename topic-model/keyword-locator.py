"""Locates the URLs for specified keywords (for easy filtering).

Usage:
  keyword-locator.py <keywords> [--head=<head>]

Options:
  -h --help      Display usage.
  --head=<head>  Limits the output. [default: 10]
"""
from docopt import docopt
from util import MongoDBLoader


if __name__ == "__main__":
    arguments = docopt(__doc__)
    MongoDBLoader().locate_keywords(arguments['<keywords>'], int(arguments['--head']))
