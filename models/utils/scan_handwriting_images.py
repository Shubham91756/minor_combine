from pathlib import Path
from PIL import Image

ROOT = Path('D:/Bunny/ML_mini_project')
DATA = ROOT / 'new_data' / 'Handwriting'

exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff'}

all_files = list(DATA.rglob('*'))
file_count = 0
image_ext_count = 0
valid_count = 0
skipped = []

for p in all_files:
    if p.is_file():
        file_count += 1
        s = p.suffix.lower()
        if s in exts:
            image_ext_count += 1
            # skip mac metadata files (trainer skips names starting with '._')
            if p.name.startswith('._'):
                skipped.append((str(p), 'mac_metadata'))
                continue
            try:
                with Image.open(p) as im:
                    im.verify()
                valid_count += 1
            except Exception as e:
                skipped.append((str(p), f'verify_fail: {e}'))
        else:
            # non-image extension
            skipped.append((str(p), 'non_image_ext'))

print('Total files under dataset:', file_count)
print('Files with image extensions:', image_ext_count)
print('Valid image files (passed verify and accepted):', valid_count)
print('Skipped files total:', len(skipped))

# show up to 40 skipped samples grouped by reason
from collections import defaultdict
by_reason = defaultdict(list)
for path, reason in skipped:
    by_reason[reason].append(path)

for reason, paths in by_reason.items():
    print('\nReason:', reason, 'count=', len(paths))
    for p in paths[:10]:
        print('  ', p)

# Exit code 0
