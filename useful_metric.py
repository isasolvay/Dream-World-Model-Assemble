import argparse
import pandas as pd
from pathlib import Path
import os

def parse_args():
    parser = argparse.ArgumentParser(description="argument parser", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--uri", default='99f26e5dc334468ba20cbaa433