"""Converts train_models.py (percent format) into Ensemble_Dashboard.ipynb"""
import re, nbformat as nbf
src = open("train_models.py", encoding="utf8").read()
chunks = re.split(r"^# %%", src, flags=re.M)[1:]
nb = nbf.v4.new_notebook(); nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
for ch in chunks:
    head, _, body = ch.partition("\n")
    if "[markdown]" in head:
        nb.cells.append(nbf.v4.new_markdown_cell("\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.strip().split("\n"))))
    else:
        nb.cells.append(nbf.v4.new_code_cell(body.strip()))
nbf.write(nb, "Ensemble_Dashboard.ipynb"); print("cells:", len(nb.cells))
