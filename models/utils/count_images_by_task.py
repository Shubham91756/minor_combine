from pathlib import Path
from PIL import Image
from collections import defaultdict

ROOT = Path('D:/Bunny/ML_mini_project')
DATA = ROOT / 'new_data' / 'Handwriting'

exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff'}

summary = defaultdict(int)
per_task = defaultdict(int)
per_class_task = defaultdict(int)
valid_images = []

for class_dir in sorted([p for p in DATA.iterdir() if p.is_dir()]):
    class_name = class_dir.name
    for task_dir in sorted([p for p in class_dir.iterdir() if p.is_dir()]):
        task_name = task_dir.name
        cnt = 0
        # walk files under task_dir
        for p in task_dir.rglob('*'):
            if p.is_file() and p.suffix.lower() in exts:
                # try to verify
                try:
                    with Image.open(p) as im:
                        im.verify()
                    cnt += 1
                    valid_images.append(str(p))
                    per_class_task[(class_name, task_name)] += 1
                    per_task[task_name] += 1
                    summary[class_name] += 1
                except Exception:
                    # skip invalid (likely mac metadata)
                    pass
        if cnt > 0:
            print(f'Class={class_name:10} Task={task_name:20} ValidImages={cnt}')

print('\nTotals by class:')
for cls, cnt in summary.items():
    print(f'  {cls:10}: {cnt}')

print('\nTotals by task across classes:')
for task, cnt in sorted(per_task.items(), key=lambda x: -x[1]):
    print(f'  {task:25}: {cnt}')

# check if any 'train' directories exist
train_dirs = list(DATA.rglob('train')) + list(DATA.rglob('Train')) + list(DATA.rglob('TRAIN'))
train_dirs = [d for d in train_dirs if d.is_dir()]
print('\nFound train-like directories:', len(train_dirs))
for d in train_dirs[:10]:
    print('  ', d)

# write validated list
out = ROOT / 'models' / 'trained_models' / 'handwriting_all_valid_images.txt'
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, 'w', encoding='utf8') as f:
    for v in valid_images:
        f.write(v + '\n')

print('\nWrote validated image list to', out)
