"""Convert a '# %%' percent-format script into an .ipynb and (optionally) execute it.

Usage: python src/py2nb.py notebooks_src/Q1_Q2_analysis.py [--run]
Markdown cells: '# %% [markdown]' followed by lines each prefixed with '# ' (or '#').
Notebooks are executed with kernel 'pa1' and cwd = <repo>/notebooks (so ../data etc. work).
"""
import sys, re, pathlib, nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell


def convert(src_path):
    text = pathlib.Path(src_path).read_text()
    chunks = re.split(r'^# %%(.*)$', text, flags=re.M)
    cells = []
    # chunks: [pre, header1, body1, header2, body2, ...]
    for header, body in zip(chunks[1::2], chunks[2::2]):
        body = body.strip('\n')
        if not body.strip():
            continue
        if 'markdown' in header:
            md = '\n'.join(re.sub(r'^# ?', '', l) for l in body.split('\n'))
            cells.append(new_markdown_cell(md))
        else:
            cells.append(new_code_cell(body))
    nb = new_notebook(cells=cells)
    nb.metadata['kernelspec'] = {'display_name': 'pa1', 'language': 'python', 'name': 'pa1'}
    return nb


if __name__ == '__main__':
    src = pathlib.Path(sys.argv[1])
    out = pathlib.Path('notebooks') / (src.stem + '.ipynb')
    nb = convert(src)
    if '--run' in sys.argv:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=None, kernel_name='pa1',
                       resources={'metadata': {'path': str(out.parent)}}).execute()
    nbformat.write(nb, out)
    print('wrote', out)
