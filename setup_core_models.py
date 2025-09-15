import os, pathlib, huggingface_hub as hf
ROOT=pathlib.Path("/data/models"); ROOT.mkdir(parents=True, exist_ok=True)
def fetch(repo_id, dst): hf.snapshot_download(repo_id, local_dir=dst, local_dir_use_symlinks=False)
fetch(os.getenv("VSRM_REPO","org/VSRM"), ROOT/"vsrm")
fetch(os.getenv("RVRT_REPO","JingyunLiang/RVRT"), ROOT/"rvrt")
fetch(os.getenv("VRT_REPO","JingyunLiang/VRT"), ROOT/"vrt")
print("Core models ready at /data/models")