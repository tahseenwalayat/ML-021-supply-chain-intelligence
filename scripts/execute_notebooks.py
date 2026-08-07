import glob
import json
import io
import contextlib
import traceback
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

nb_files = sorted(glob.glob("notebooks/*.ipynb"))

for nb_path in nb_files:
    print(f"Executing {nb_path}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    exec_globals = {}

    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            code = "".join(cell["source"])
            output_buffer = io.StringIO()

            try:
                with contextlib.redirect_stdout(output_buffer):
                    exec(code, exec_globals)
                out_str = output_buffer.getvalue()
                cell["outputs"] = [{
                    "name": "stdout",
                    "output_type": "stream",
                    "text": out_str.splitlines(True)
                }]
                cell["execution_count"] = idx
            except Exception as e:
                err_msg = f"Error in cell {idx}: {e}\n{traceback.format_exc()}"
                print(err_msg)
                cell["outputs"] = [{
                    "ename": type(e).__name__,
                    "evalue": str(e),
                    "output_type": "error",
                    "traceback": traceback.format_exc().splitlines(True)
                }]

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

print("All notebooks executed and outputs recorded successfully!")
