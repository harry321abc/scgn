#!/usr/bin/env python
# -*- encoding: utf-8 -*-



import argparse
import configparser
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', nargs='+', type=int)
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--exclude_radicals', type=bool, default=False)
    args = parser.parse_args()
    old_cfg = configparser.ConfigParser()
    new_cfg = configparser.ConfigParser()
    old_cfg.read(args.input)
    i = 0
    for index in args.index:
        reaction = old_cfg.get('reaction_%d' % index, 'reaction')
        match_result = re.findall('(J)', reaction.split('>')[0])
        if args.exclude_radicals and len(match_result) > 1:
            continue
        new_cfg.add_section('reaction_%d' % (i + 1))
        for k, v in old_cfg.items('reaction_%d' % index):
            new_cfg.set('reaction_%d' % (i + 1), k, v)
        i += 1
    new_cfg.write(open(args.output, 'w'))


if __name__ == '__main__':
    main()
