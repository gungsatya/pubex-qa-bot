from huggingface_hub import hf_hub_download, list_repo_files

REPO = "Qwen/Qwen3-VL-2B-Instruct-GGUF"

# pilih file yang kamu mau (umumnya aman: language Q4_K_M + mmproj Q8_0 atau F16)
files = list_repo_files(REPO)

# contoh pemilihan sederhana berbasis nama file
model_file = next(f for f in files if "Q4_K_M" in f and f.endswith(".gguf") and "mmproj" not in f)
mmproj_file = next(f for f in files if "mmproj" in f and ("Q8_0" in f or "F16" in f) and f.endswith(".gguf"))

model_path = hf_hub_download(REPO, filename=model_file)
mmproj_path = hf_hub_download(REPO, filename=mmproj_file)

print("model:", model_path)
print("mmproj:", mmproj_path)
