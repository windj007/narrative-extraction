#!/usr/bin/env python3
import os

import requests

import yandex_search

from narratex.logger import setup_logger


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    logger = setup_logger()

    yandex = yandex_search.Yandex(api_user=args.user, api_key=args.key)
    with open(args.queries_file, 'r') as queries_f:
        for query_i, query in enumerate(queries_f):
            query = query.strip()
            if not query:
                continue

            logger.info(f'Query {query_i}: {query}')
            query_res_i = 0
            for page_i in range(args.get_pages):
                for found_item in yandex.search(query, page=page_i).items:
                    url = found_item['url']
                    logger.info(f'Found item {query_res_i}: {url}')

                    resp = requests.get(url)
                    with open(os.path.join(args.out_dir, f'{query_i:03d}_{query_res_i:05d}.html'), 'w') as item_f:
                        item_f.write(resp.content)

                    query_res_i += 1


if __name__ == '__main__':
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument('user', type=str, help='Username to use in Yandex API')
    aparser.add_argument('key', type=str, help='Key to use in Yandex API')
    aparser.add_argument('queries_file', type=str, help='Path to file with queries, one query per line')
    aparser.add_argument('out_dir', type=str, help='Where to store results')
    aparser.add_argument('--get-pages', type=int, default=3, help='How many pages to get')

    main(aparser.parse_args())
