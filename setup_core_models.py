
"""
MIT License

Copyright (c) 2024 Video Enhancement Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os, pathlib, huggingface_hub as hf
ROOT=pathlib.Path("/data/models"); ROOT.mkdir(parents=True, exist_ok=True)
def fetch(repo_id, dst): hf.snapshot_download(repo_id, local_dir=dst, local_dir_use_symlinks=False)
fetch(os.getenv("VSRM_REPO","org/VSRM"), ROOT/"vsrm")
fetch(os.getenv("RVRT_REPO","JingyunLiang/RVRT"), ROOT/"rvrt")
fetch(os.getenv("VRT_REPO","JingyunLiang/VRT"), ROOT/"vrt")
print("Core models ready at /data/models")