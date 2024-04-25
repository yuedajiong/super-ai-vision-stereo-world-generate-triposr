1. modified from https://github.com/VAST-AI-Research/TripoSR
2. wget https://huggingface.co/stabilityai/TripoSR/blob/main/model.ckpt ./ckpt/TripoSR/model.ckpt
3. put your .obj files under ./data/image/resin/{lion}/, and run .py under ./code/: superv_01_data_01_mesh.py -> superv_01_data_02_focus.py -> superv_01_data_03_split.py
   OR: direct unzip data.zip
4. check code in superv.py, commnet or uncomment 'infer/train' lines.  (the infer code is fully compatible to official TripoSR,  the tarin code is configured for small network.)

#logic correct, no distributed version, no large scale train. 

have fun.
