"""Versioned JSON-lines protocol. stdout contains responses only."""
import argparse
import json
from pathlib import Path
import sys

from .app import AppService


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    service = AppService(args.data_dir)
    try:
        for line in sys.stdin:
            request = {}
            try:
                request = json.loads(line)
                if not isinstance(request, dict) or request.get('version') != 1:
                    raise ValueError('Unsupported application protocol')
                result = service.call(request['method'], request.get('params', {}))
                response = dict(id=request.get('id'), result=result)
            except Exception as error:
                response = dict(id=request.get('id') if isinstance(request, dict) else None,
                                error=str(error))
            print(json.dumps(response), flush=True)
    finally:
        service.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
