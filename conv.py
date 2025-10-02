import nbformat as nbf

# Читаем весь код из файла
with open("solution.py", "r", encoding="utf-8") as f:
    code = f.read()

# Разбиваем по ячейкам (ищем маркеры "# %%")
cells = []
for block in code.split("# %%"):
    block = block.strip()
    if not block:
        continue
    if block.startswith("[markdown]"):
        # markdown-ячейка
        cells.append(nbf.v4.new_markdown_cell(block.replace("[markdown]", "").strip()))
    else:
        # code-ячейка
        cells.append(nbf.v4.new_code_cell(block))

nb = nbf.v4.new_notebook(cells=cells)

# Сохраняем
with open("solution.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
