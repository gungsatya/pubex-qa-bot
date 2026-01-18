from litepali import LitePali

litepali = LitePali(device="cpu")
litepali.load_index("./src/data/indexes/litepali_index")

results = litepali.search("akusisi dari perusahaan ABM", k=5)
for r in results:
    print(r["score"], r["image"].path, r["image"].metadata)
