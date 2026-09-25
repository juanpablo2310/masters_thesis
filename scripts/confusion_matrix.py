"""
Styled Confusion Matrix Generator with Image Labels
=====================================================
Generates publication-quality confusion matrices where axis labels
can be either text strings or PNG image paths.
 
Usage:
    python confusion_matrix.py
 
Customize the DATA and LABELS sections at the bottom of this file.
"""
 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from PIL import Image
from scipy.ndimage import rotate as ndimage_rotate
import os
 
 
# ─── STYLE CONSTANTS ──────────────────────────────────────────────────────────
 
FONT_FAMILY     = "Georgia"          # serif feels more editorial / scientific
BG_COLOR        = "#F7F5F0"          # warm off-white background
GRID_COLOR      = "#FFFFFF"          # white cell separators
TITLE_COLOR     = "#1A1A2E"          # deep navy title
TICK_COLOR      = "#3D3D3D"          # dark-grey tick labels
VALUE_COLOR_LO  = "#1A1A2E"         # text color on light cells
VALUE_COLOR_HI  = "#FFFFFF"          # text color on dark cells
 
# Custom colormap: pale cream → deep indigo
CMAP = LinearSegmentedColormap.from_list(
    "indigo_cream",
    ["#EAE6DF", "#C9D4E8", "#7B9BC8", "#2E5FA3", "#1A2E6E"],
    N=256,
)
 
# ─── CORE FUNCTION ────────────────────────────────────────────────────────────
 
def draw_image_label(
    ax,
    img_path: str,
    xy,
    zoom=1.0,
    rotation=0,
    target_size=(52, 52),  # <- tamaño uniforme pequeño pero legible
):
    """
    Draw image labels with a fixed visual size.
    """

    img = Image.open(img_path).convert("RGBA")

    # Mantener aspect ratio dentro del tamaño fijo
    img.thumbnail(target_size, Image.Resampling.LANCZOS)

    # Crear canvas transparente fijo
    canvas = Image.new("RGBA", target_size, (0, 0, 0, 0))

    # Centrar imagen en canvas
    offset_x = (target_size[0] - img.width) // 2
    offset_y = (target_size[1] - img.height) // 2
    canvas.paste(img, (offset_x, offset_y), img)

    img = np.array(canvas)

    # Rotación
    if rotation != 0:
        img = ndimage_rotate(img, rotation, reshape=True, cval=0)

    imagebox = OffsetImage(img, zoom=zoom)

    imagebox.image.axes = ax

    ab = AnnotationBbox(
        imagebox,
        xy,
        xycoords="axes fraction",
        frameon=False,
        pad=0,
        box_alignment=(0.5, 0.5),
    )

    ax.add_artist(ab)
 
 
def plot_confusion_matrix(
    matrix: np.ndarray,
    labels,                   # list of str  OR  list of image paths (str/Path)
    title: str = "Confusion Matrix Normalized",
    figsize: tuple = None,
    image_zoom: float = 0.55,
    save_path: str = None,
    show: bool = True,
):
    """
    Parameters
    ----------
    matrix      : 2-D numpy array, values in [0, 1] (normalized).
    labels      : list of class names (str) OR paths to PNG images.
                  Mix is supported — each entry is treated as an image
                  path if the string ends in a recognised image extension
                  and the file exists; otherwise as plain text.
    title       : Figure title.
    figsize     : (width, height) in inches.  Auto-sized if None.
    image_zoom  : Zoom factor applied to label images.
    save_path   : If given, figure is saved to this path (e.g. "output.png").
    show        : Whether to call plt.show().
    """
    n = len(labels)
    if figsize is None:
        base = max(7, n * 1.15)
        figsize = (base + 2.5, base)
 
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
 
    # ── draw heatmap ──────────────────────────────────────────────────────────
    im = ax.imshow(matrix, cmap=CMAP, vmin=0, vmax=matrix.max(), aspect="equal")
 
    # thin white grid lines between cells
    for x in np.arange(-0.5, n, 1):
        ax.axhline(x, color=GRID_COLOR, linewidth=0.8, zorder=2)
        ax.axvline(x, color=GRID_COLOR, linewidth=0.8, zorder=2)
 
    # ── cell annotations ──────────────────────────────────────────────────────
    threshold = matrix.max() * 0.55
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = VALUE_COLOR_HI if val >= threshold else VALUE_COLOR_LO
            weight = "bold" if i == j else "normal"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=9.5, color=color,
                fontweight=weight,
                fontfamily=FONT_FAMILY,
                zorder=3,
            )
 
    # ── colorbar ──────────────────────────────────────────────────────────────
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=8, colors=TICK_COLOR)
    cbar.outline.set_edgecolor("#CCCCCC")
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(0.5)
    cbar.ax.yaxis.set_tick_params(color=TICK_COLOR)
 
    # ── detect whether labels are images ──────────────────────────────────────
    _img_exts = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}
 
    def _is_image(lbl):
        p = Path(str(lbl))
        return p.suffix.lower() in _img_exts and p.exists()
 
    use_images = [_is_image(lbl) for lbl in labels]
 
    # ── text tick labels (shown even when images are used, as fallback) ───────
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
 
    # if ALL labels are images, hide the text ticks; otherwise show them
    if all(use_images):
        ax.set_xticklabels([""] * n)
        ax.set_yticklabels([""] * n)
    else:
        ax.set_xticklabels(
            [str(lbl) if not use_images[i] else "" for i, lbl in enumerate(labels)],
            rotation=45, ha="right", fontsize=9,
            color=TICK_COLOR, fontfamily=FONT_FAMILY,
        )
        ax.set_yticklabels(
            [str(lbl) if not use_images[i] else "" for i, lbl in enumerate(labels)],
            fontsize=9, color=TICK_COLOR, fontfamily=FONT_FAMILY,
        )
 
    ax.tick_params(axis="both", which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
 
    # ── image labels ──────────────────────────────────────────────────────────
    # We place images outside the axes using figure coordinates.
    # Layout: compute where each cell centre maps to in axes fraction,
    # then nudge a bit further outside.
 
    if any(use_images):
        fig.canvas.draw()          # needed to flush layout before placing images
        renderer = fig.canvas.get_renderer()
        ax_bbox = ax.get_window_extent(renderer)   # pixels
 
        # cell size in axes fraction
        cell_frac_x = 1.0 / n
        cell_frac_y = 1.0 / n
 
        # offset (in axes fraction) to push labels outside the plot area
        x_offset_frac = cell_frac_x * 0.48   # below x-axis
        y_offset_frac = cell_frac_y * 0.48   # left of y-axis
 
        for idx, lbl in enumerate(labels):
            if not use_images[idx]:
                continue
 
            # cell centre in axes fraction
            cx = (idx + 0.5) / n
            cy = (idx + 0.5) / n   # row idx → y runs top-to-bottom in imshow
 
            # x-axis label (below the axes) — rotated 90° so it reads vertically
            draw_image_label(
                ax, str(lbl),
                xy=(cx, -x_offset_frac),
                zoom=image_zoom,
                rotation=90,
            )
 
            # y-axis label (left of the axes) — horizontal (no rotation)
            draw_image_label(
                ax, str(lbl),
                xy=(-y_offset_frac, 1.0 - cy),   # imshow flips y
                zoom=image_zoom,
                rotation=0,
            )
 
    # ── axis titles ───────────────────────────────────────────────────────────
    label_pad = 55 if any(use_images) else 15
    ax.set_xlabel(
        "True", fontsize=12, labelpad=label_pad,
        color=TITLE_COLOR, fontfamily=FONT_FAMILY, fontweight="bold",
    )
    ax.set_ylabel(
        "Predicted", fontsize=12, labelpad=label_pad,
        color=TITLE_COLOR, fontfamily=FONT_FAMILY, fontweight="bold",
    )
 
    # ── title ─────────────────────────────────────────────────────────────────
    ax.set_title(
        title,
        fontsize=16, pad=18,
        color=TITLE_COLOR,
        fontfamily=FONT_FAMILY,
        fontweight="bold",
        loc="center",
    )
 
    # ── diagonal accuracy strip ───────────────────────────────────────────────
    # Subtle highlight box in the top-right corner showing mean diagonal acc.
    mean_acc = np.mean(np.diag(matrix))
    fig.text(
        0.97, 0.97,
        f"Mean accuracy\n{mean_acc:.1%}",
        ha="right", va="top",
        fontsize=8.5, color=TICK_COLOR,
        fontfamily=FONT_FAMILY,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#FFFFFF",
            edgecolor="#CCCCCC",
            linewidth=0.8,
            alpha=0.85,
        ),
    )
 
    plt.tight_layout(pad=2.5)
 
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=BG_COLOR)
        print(f"Saved → {save_path}")
 
    if show:
        plt.show()
 
    return fig, ax


# ─── EXAMPLE DATA ─────────────────────────────────────────────────────────────
# Replace these matrices / labels with your own.

if __name__ == "__main__":
    
    BASE_PATH = "/Volumes/ADATA HD680/Shared/Files From d.localized/Maestria/tesis/herbario/results_analysis/images/"

    # ── Example 1: Matrix 1 (Mascara classes) with TEXT labels ────────────────
    matrix_1 = np.array([
    [0.64, 0.07, 0.06, 0.07, 0.06, 0.04, 0.07],
    [0.01, 0.82, 0.00, 0.03, 0.04, 0.02, 0.08],
    [0.03, 0.02, 0.78, 0.01, 0.07, 0.01, 0.08],
    [0.06, 0.02, 0.00, 0.75, 0.06, 0.06, 0.06],
    [0.01, 0.03, 0.01, 0.08, 0.84, 0.03, 0.01],
    [0.03, 0.03, 0.06, 0.05, 0.07, 0.75, 0.01],
    [0.05, 0.06, 0.04, 0.06, 0.04, 0.04, 0.71]
]
)

    labels_1 = [
        os.path.join(BASE_PATH,'UNAL' ,"Mascara_Codigos.jpg"),
        os.path.join(BASE_PATH,'UNAL' ,"Mascara_ColorChecker.jpg"),
        os.path.join(BASE_PATH,'UNAL' ,"Mascara_Descripciones.jpg"),
        os.path.join(BASE_PATH,'UNAL' ,"Mascara_Encabezados.jpg"),
        os.path.join(BASE_PATH,'UNAL' ,"Mascara_Escalas.jpg"),
        os.path.join(BASE_PATH,'UNAL' ,"Mascara_Sellos.jpg"),
        "background", 
    ]

    # ── Example 2: Matrix 2 (Melu classes) with TEXT labels ───────────────────
    matrix_2 = np.array([
    [0.45, 0.02, 0.08, 0.08, 0.08, 0.04, 0.05, 0.01, 0.06, 0.02, 0.08, 0.04],
    [0.04, 0.49, 0.04, 0.08, 0.04, 0.06, 0.02, 0.08, 0.05, 0.02, 0.04, 0.07],
    [0.04, 0.03, 0.62, 0.03, 0.09, 0.08, 0.04, 0.02, 0.02, 0.00, 0.00, 0.03],
    [0.01, 0.05, 0.04, 0.49, 0.04, 0.06, 0.03, 0.02, 0.07, 0.06, 0.08, 0.07],
    [0.08, 0.05, 0.02, 0.09, 0.54, 0.03, 0.03, 0.08, 0.03, 0.01, 0.02, 0.03],
    [0.07, 0.02, 0.01, 0.04, 0.07, 0.49, 0.06, 0.06, 0.03, 0.05, 0.07, 0.03],
    [0.02, 0.04, 0.07, 0.05, 0.03, 0.02, 0.50, 0.04, 0.08, 0.04, 0.07, 0.04],
    [0.02, 0.07, 0.06, 0.08, 0.02, 0.07, 0.04, 0.47, 0.04, 0.04, 0.04, 0.04],
    [0.02, 0.06, 0.08, 0.08, 0.02, 0.02, 0.08, 0.02, 0.50, 0.07, 0.03, 0.02],
    [0.04, 0.04, 0.00, 0.07, 0.03, 0.07, 0.07, 0.03, 0.01, 0.56, 0.04, 0.03],
    [0.09, 0.08, 0.05, 0.02, 0.06, 0.01, 0.09, 0.00, 0.02, 0.03, 0.53, 0.01],
    [0.00, 0.05, 0.06, 0.03, 0.05, 0.09, 0.02, 0.03, 0.08, 0.01, 0.00, 0.59]
])

    labels_2 = [
        os.path.join(BASE_PATH,"MELU","small database label.jpg"),
        os.path.join(BASE_PATH,"MELU","handwritten data.jpg"),
        os.path.join(BASE_PATH,"MELU","stamp.jpg"),
        os.path.join(BASE_PATH,"MELU","annotation label.jpg"),
        os.path.join(BASE_PATH,"MELU","scale.jpg"),
        os.path.join(BASE_PATH,"MELU","swing tag.jpg"),
        os.path.join(BASE_PATH,"MELU","full database label.jpg"),
        os.path.join(BASE_PATH,"MELU","database label.jpg"),
        os.path.join(BASE_PATH,"MELU","swatch.jpg"),
        os.path.join(BASE_PATH,"MELU","institutional label.jpg"),
        os.path.join(BASE_PATH,"MELU","number.jpg"),
        "background",
    ]

    # ─── TO USE IMAGE LABELS instead of text, replace the strings above
    # ─── with file paths to your PNG icons, e.g.:
    #
    #   labels_1 = [
    #       "icons/codigos.png",
    #       "icons/colorchecker.png",
    #       ...
    #   ]
    #
    # ─── You can also mix text and image labels in the same list.

    plot_confusion_matrix(
        matrix_1, labels_1,
        title="Confusion Matrix Normalized — UNAL",
        save_path="lone_confusion_matrix_unal.png",
        show=True,
    )

    plot_confusion_matrix(
        matrix_2, labels_2,
        title="Confusion Matrix Normalized — MELU",
        figsize=(14, 12),
        save_path="lone_confusion_matrix_melu.png",
        show=True,
    )
