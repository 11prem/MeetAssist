"""Dev-time helper: download a model repo from ModelScope (HF is blocked on the work network).
usage: python ms_download.py <owner/name> <dest_dir> [--list] [--only substr]
Not part of the shipped app (the app itself never touches the network except the LLM)."""
import sys, os, requests
try:
    import truststore; truststore.inject_into_ssl()
except Exception: pass
B = "https://www.modelscope.cn/api/v1/models/"
def files(mid):
    r = requests.get(f"{B}{mid}/repo/files", params={"Recursive": "true", "Revision": "master"}, timeout=60); r.raise_for_status()
    return [f for f in r.json()["Data"]["Files"] if f["Type"] == "blob"]
def main():
    mid, dest = sys.argv[1], sys.argv[2]; only = sys.argv[sys.argv.index("--only")+1] if "--only" in sys.argv else None
    fl = files(mid)
    if "--list" in sys.argv:
        for f in fl: print(f"{f['Size']/1e6:10.1f} MB  {f['Path']}")
        return
    for f in fl:
        if only and only not in f["Path"]: continue
        out = os.path.join(dest, f["Path"]); os.makedirs(os.path.dirname(out), exist_ok=True)
        if os.path.exists(out) and os.path.getsize(out) == f["Size"]: continue
        print("get", f["Path"], round(f["Size"]/1e6,1), "MB", flush=True)
        with requests.get(f"{B}{mid}/repo", params={"Revision": "master", "FilePath": f["Path"]}, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(out + ".part", "wb") as fh:
                for ch in r.iter_content(1 << 20): fh.write(ch)
        os.replace(out + ".part", out)
main()
