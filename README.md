```bash
conda create -n smiles python=3.10 --no-default-packages -y
conda activate smiles
pip install uv
uv pip install -r requirements.txt
```