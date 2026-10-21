"""Make lm/data/corpus.txt, the byte corpus lab.py trains on (first 90% train, last 10% val).

python3 prepare_data.py                      downloads public-domain books from Project Gutenberg (up to 8.4 MB)
python3 prepare_data.py --from a.txt b.md    concatenates your own text files instead
The results in RESULTS.md used an 8.4 MB English text corpus that is not released, so absolute losses on this
corpus will differ; the per-head vs Muon comparison is what the scripts test.
"""
import argparse, os, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BOOKS = [1342, 84, 11, 1661, 2701, 98, 1952, 345, 174, 2600]  # Pride and Prejudice, Frankenstein, Alice, Sherlock Holmes, Moby Dick, ...
MB = 8.4

ap = argparse.ArgumentParser()
ap.add_argument("--from", dest="files", nargs="*", default=None)
a = ap.parse_args()
if a.files:
    text = b"\n\n".join(open(f, "rb").read() for f in a.files)
else:
    parts = []
    for b in BOOKS:
        url = f"https://www.gutenberg.org/cache/epub/{b}/pg{b}.txt"
        print("downloading", url); parts.append(urllib.request.urlopen(url).read())
        if sum(map(len, parts)) > MB * 1e6: break
    text = b"\n\n".join(parts)[:int(MB * 1e6)]
os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
open(os.path.join(HERE, "data", "corpus.txt"), "wb").write(text)
print(f"wrote data/corpus.txt, {len(text) / 1e6:.1f} MB")
