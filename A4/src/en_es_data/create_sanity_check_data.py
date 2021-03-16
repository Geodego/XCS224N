#!/usr/bin/env python3

import random


def create_sanity_file(t):
    en_file = t[0]
    es_file = t[1]
    category = t[2]
    en_subset, es_subset = sample_rows_from_file(en_file, es_file)
    write_sanity_check_file(en_subset, '{}_sanity_check.en'.format(category))
    write_sanity_check_file(es_subset, '{}_sanity_check.es'.format(category))


def sample_rows_from_file(en_file, es_file):
    with open(en_file) as f:
        en_lines = f.readlines()
    with open(es_file) as f:
        es_lines = f.readlines()
    comb = list(zip(en_lines, es_lines))
    random.shuffle(comb)
    en_subset, es_subset = zip(*comb)
    return en_subset, es_subset


def write_sanity_check_file(data, file_name):
    if 'dev' in file_name:
        data = data[:25]
    elif 'train' in file_name:
        data = data[:50]
    elif 'test' in file_name:
        data = data[:5]

    with open(file_name, 'w') as f:
        f.writelines(data)


if __name__ == '__main__':
    files = [
        ('dev.en', 'dev.es', 'dev'),
        ('test.en', 'test.es', 'test'),
        ('train.en', 'train.es', 'train')
    ]
    for t in files:
        create_sanity_file(t)
