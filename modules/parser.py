import argparse

def parse_option():
    parser = argparse.ArgumentParser(description='Run model with weight option.')
    parser.add_argument('--imagenet', action=argparse.BooleanOptionalAction, help='Do you need to use imagenet weight (pretrained)')
    return parser.parse_args()
