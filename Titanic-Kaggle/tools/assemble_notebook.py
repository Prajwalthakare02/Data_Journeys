# Assemble Titanic.ipynb from the nb_src/ fragment files.
# *.py fragments become code cells, *.md fragments become markdown cells,
# ordered by filename (numeric prefixes enforce the teaching sequence).
import json
import os

SRC = 'nb_src'
OUT = 'Titanic.ipynb'

cells = []
for fname in sorted(os.listdir(SRC)):
    if not os.path.isfile(os.path.join(SRC, fname)):
        continue
    path = os.path.join(SRC, fname)
    with open(path, encoding='utf-8') as fh:
        src = fh.read().rstrip('\n')
    lines = src.split('\n')
    # Jupyter cell 'source' wants a list of NEWLINE-TERMINATED strings
    source = [ln + '\n' for ln in lines[:-1]]
    if lines[-1]:
        source.append(lines[-1])
    if fname.endswith('.md'):
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': source})
    elif fname.endswith('.py'):
        cells.append({'cell_type': 'code', 'execution_count': None,
                      'metadata': {}, 'outputs': [], 'source': source})
    else:
        raise ValueError(f'unhandled fragment type: {fname}')

nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12.0'},
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}

with open(OUT, 'w', encoding='utf-8') as fh:
    json.dump(nb, fh, indent=1, ensure_ascii=False)

print(f'assembled {len(cells)} cells -> {OUT}')