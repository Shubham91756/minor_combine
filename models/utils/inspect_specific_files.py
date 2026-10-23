from pathlib import Path
from PIL import Image

paths = [
    Path(r"D:/Bunny/ML_mini_project/new_data/Handwriting/healthy/Healthy Circle/__MACOSX/Circle/._circA-P28.jpg"),
    Path(r"D:/Bunny/ML_mini_project/new_data/Handwriting/healthy/Healthy Circle/__MACOSX/Circle/._circA-P29.jpg"),
]

for p in paths:
    print('\nInspecting:', p)
    print('Exists:', p.exists())
    if p.exists():
        try:
            sz = p.stat().st_size
        except Exception as e:
            sz = f'error getting size: {e}'
        print('Size (bytes):', sz)
        try:
            with p.open('rb') as f:
                data = f.read(32)
            print('First 32 bytes (hex):', data.hex())
        except Exception as e:
            print('Could not read bytes:', e)
        try:
            with Image.open(p) as im:
                print('Pillow format:', im.format, 'size:', im.size, 'mode:', im.mode)
        except Exception as e:
            print('Pillow open failed:', e)

# Also check the corresponding real images (without '._' prefix) if they exist
base_dir = Path(r"D:/Bunny/ML_mini_project/new_data/Handwriting/healthy/Healthy Circle/Circle")
for name in ['circA-P28.jpg', 'circA-P29.jpg']:
    p = base_dir / name
    print('\nInspecting real image candidate:', p)
    print('Exists:', p.exists())
    if p.exists():
        try:
            print('Size (bytes):', p.stat().st_size)
            with p.open('rb') as f:
                print('First 16 bytes (hex):', f.read(16).hex())
            with Image.open(p) as im:
                print('Pillow format:', im.format, 'size:', im.size, 'mode:', im.mode)
        except Exception as e:
            print('Error reading real image:', e)
