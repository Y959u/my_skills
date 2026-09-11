"""Plot every sample in a spectral CSV as two styled PNGs."""
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.ticker import FuncFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--title', default='所有样本的光谱反射率')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--font', type=Path)
    args = parser.parse_args()
    with args.csv.open(encoding='utf-8-sig', newline='') as handle:
        rows = [r for r in csv.reader(handle) if r]
    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError('CSV must contain wavelength headers and at least one sample.')
    x = np.array([float(h.removeprefix('B_')) for h in rows[0][1:]])
    if any(len(r) != len(rows[0]) for r in rows[1:]):
        raise ValueError('CSV rows have inconsistent column counts.')
    y = np.array([[float(v) for v in r[1:]] for r in rows[1:]])
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Missing or nonfinite spectral values are not supported.')
    if len(np.unique(x)) != len(x):
        raise ValueError('Duplicate wavelengths found.')
    font_path = args.font
    if font_path is None:
        windows_font = Path('C:/Windows/Fonts/msyh.ttc')
        if windows_font.exists():
            font_path = windows_font
        else:
            font_path = next((Path(f.fname) for f in fontManager.ttflist
                              if f.name in ('Noto Sans CJK SC', 'SimHei', 'Microsoft YaHei')), None)
    if font_path is None or not font_path.is_file():
        raise ValueError('Specify a Chinese-capable font file with --font.')
    font = FontProperties(fname=str(font_path))
    order = np.argsort(x)
    colors = np.random.default_rng(args.seed).choice(
        ['#2878B5', '#2A9D65', '#E68A2E', '#8B5BB5'], size=len(y))
    output = args.output_dir or args.csv.resolve().parent
    output.mkdir(parents=True, exist_ok=True)
    right = max(1020, float(x.max()) + 20)
    ticks = np.arange(np.ceil(x.min()/100)*100, right + 1, 100)
    for has_title, suffix in [(True, 'with_title'), (False, 'without_title')]:
        fig, ax = plt.subplots(figsize=(10, 6), layout='constrained', facecolor='white')
        ax.set_facecolor('white')
        for values, color in zip(y, colors):
            ax.plot(x[order], values[order], color=color, lw=.75, alpha=.7)
        ax.set_xlabel('波长（nm）', fontproperties=font, fontsize=13)
        ax.set_ylabel('反射率', fontproperties=font, fontsize=13)
        if has_title:
            ax.set_title(args.title, fontproperties=font, fontsize=16, pad=15)
        ax.set_xlim(x.min(), right)
        ax.set_xticks(ticks)
        low, high = min(0, float(y.min())), max(0, float(y.max()))
        padding = (high-low)*.05 or .05
        ax.set_ylim(low-padding if low < 0 else low, high+padding)
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda value, pos: '' if abs(value) < 1e-10 else f'{value:g}'))
        ax.grid(False)
        ax.tick_params(labelsize=11, direction='in')
        ax.spines[['top', 'right']].set_visible(False)
        target = output / f'all_samples_reflectance_{suffix}.png'
        fig.savefig(target, dpi=300, facecolor='white')
        plt.close(fig)
        print(target)
    print(f'Samples: {len(y)}; bands: {len(x)}; wavelength range: {x.min()}–{x.max()} nm')


if __name__ == '__main__':
    main()
