import zipfile

zip_path = "/data/npl/ViInfographicCaps/Contest/final_contest/another_way/stanford-corenlp-4.5.6.zip"
extract_dir = "/data/npl/ViInfographicCaps/Contest/final_contest/another_way/"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("✅ Unzipped successfully.")