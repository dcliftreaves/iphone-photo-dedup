"""Small, literal output templates: no expressions, absolute paths, or traversal."""
from pathlib import Path
import re
import string

DEFAULT_TEMPLATE = '{camera}/{capture_month}/{stem}{ext}'
TOKENS = {'camera', 'capture_month', 'year', 'month', 'day', 'date', 'stem', 'ext', 'kind'}


def validate_template(template):
    if not isinstance(template, str) or not template or len(template) > 400:
        raise ValueError('Choose an output template of 1–400 characters')
    if '\\' in template or ':' in template or template.startswith('/'):
        raise ValueError('Output templates must be relative paths using / between folders')
    if any(part in {'', '.', '..'} for part in template.split('/')):
        raise ValueError('Output templates cannot contain empty folders, . or ..')
    try:
        fields = []
        for literal, field, spec, conversion in string.Formatter().parse(template):
            if re.search(r'[<>|?*\x00-\x1f]', literal):
                raise ValueError('Output template contains an unsupported filename character')
            if field is not None:
                if field not in TOKENS or spec or conversion:
                    raise ValueError('Use only these tokens: ' + ', '.join(sorted(TOKENS)))
                fields.append(field)
    except (KeyError, IndexError) as error:
        raise ValueError('Invalid output template') from error
    if fields.count('ext') != 1 or not template.endswith('{ext}') or 'stem' not in fields:
        raise ValueError('Include {stem} and end the filename with {ext} (which includes the dot)')
    if '{stem}' not in template.split('/')[-1]:
        raise ValueError('Put {stem} in the filename, not a folder')
    return template


def render_template(record, template=DEFAULT_TEMPLATE):
    from .export import safe_component
    validate_template(template)
    match = re.match(r'^(\d{4})[:-](\d{2})[:-](\d{2})', record.get('captured_at') or '')
    year, month, day = match.groups() if match else ('Unknown-year', 'Unknown-month', 'Unknown-day')
    values = dict(camera=safe_component(record.get('model', 'Unknown')),
                  capture_month=f'{year}-{month}' if match else 'Unknown-date',
                  year=year, month=month, day=day,
                  date=f'{year}-{month}-{day}' if match else 'Unknown-date',
                  stem=safe_component(Path(record['path']).stem), ext=record['extension'],
                  kind=safe_component(record.get('kind', 'Unknown')))
    parts = template.format(**values).split('/')
    extension = values['ext']
    return Path(*(safe_component(part) for part in parts[:-1]), safe_component(parts[-1][:-len(extension)]) + extension)
