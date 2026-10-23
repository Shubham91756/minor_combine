from pathlib import Path
from PIL import Image

ROOT = Path('D:/Bunny/ML_mini_project')
DATA = ROOT / 'new_data' / 'Handwriting'
OUT = ROOT / 'models' / 'trained_models' / 'handwriting_all_valid_images.txt'

exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff'}

all_files = list(DATA.rglob('*'))
image_files = [p for p in all_files if p.is_file() and p.suffix.lower() in exts]
print('Found image-extension files:', len(image_files))

valid = []
skipped = []

for p in image_files:
    try:
        with Image.open(p) as im:
            im.verify()
        valid.append(str(p))
    except Exception as e:
        skipped.append((str(p), str(e)))

print('Valid images:', len(valid))
print('Invalid/failed images:', len(skipped))

# Save the list of valid images
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, 'w', encoding='utf8') as f:
    for v in valid:
        f.write(v + '\n')

print('Wrote valid image list to', OUT)

# Print a few samples from valid and skipped
print('\nSample valid (first 20):')
for v in valid[:20]:
    print('  ', v)

print('\nSample skipped (first 20):')
for s in skipped[:20]:
    print('  ', s[0], '|', s[1])
