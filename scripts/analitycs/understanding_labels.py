#!/usr/bin/env python
# coding: utf-8


import os
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw
from tqdm import tqdm

pd.options.mode.chained_assignment = None

sns.set_theme()


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


def isValid(*dirs):
    return os.path.isfile(os.path.join(os.getcwd(), *dirs))


train["valid_path"] = [
    isValid("data", "imagenes", filename) for filename in train.filename
]
test["valid_path"] = [
    isValid("data", "imagenes", filename) for filename in test.filename
]

data_total = pd.concat([train, test]).reset_index(drop=True)
data_total.to_csv("etiquetas.csv", header=True, index_label=False, index=False)
sns.histplot(data=data_total, y="class")
plt.tight_layout()
plt.savefig("conteo_labels_dataset.png")


data_total["valid_path"] = [
    isValid("data", "imagenes", filename) for filename in data_total.filename
]


enriching_list = []
# os.makedirs()
for name in tqdm(data_total.filename.unique(), total=len(data_total.filename.unique())):
    image_file = Image.open(os.path.join(os.getcwd(), "data", "imagenes", name))
    image_df = data_total.loc[data_total.filename == name]
    image_df["Numero de etiquetas diferentes en la imagen"] = image_df.shape[0]
    image_df["Numero de clases de etiquetas diferentes en la imagen"] = len(
        image_df["class"].unique()
    )
    image_df["duplicamiento de etiqueta en x"] = image_df["xmax"].duplicated(
        keep=False
    ) & image_df["xmin"].duplicated(keep=False)
    image_df["duplicamiento de etiqueta en y"] = image_df["ymax"].duplicated(
        keep=False
    ) & image_df["ymin"].duplicated(keep=False)
    for i, row in image_df.iterrows():
        if not os.path.exists(os.path.join(os.getcwd(), row["class"])):
            os.makedirs(os.path.join(os.getcwd(), row["class"]))

        draw_row = ImageDraw.Draw(image_file)
        draw_row.rectangle(
            (row["xmin"], row["ymin"], row["xmax"], row["ymax"]), outline=(250, 0, 0)
        )
        draw_row.text((row["xmin"], row["ymin"]), row["class"], (150, 150, 150))
        label_mask = image_file.crop(
            (row["xmin"], row["ymin"], row["xmax"], row["ymax"])
        )
        label_mask.save(
            os.path.join(
                os.getcwd(), row["class"], f'{name[:-4]}_{row["class"]}_{i}.jpg'
            ),
            format="JPEG",
            quality=90,
        )
    # image_file.save(
    #     os.path.join(os.getcwd(), "label_data", f"{name[:-4]}_highligted.jpg"),
    #     format="JPEG",
    #     quality=90,
    # )
    enriching_list.append(image_df)
data_total_enrich = pd.concat(enriching_list)

# shutil.make_archive("label_data", "zip", os.path.join(os.getcwd(), "label_data"))

# print(data_total_enrich.mean(numeric_only=True).to_frame())
cols_to_select = data_total_enrich.drop(
    ["xmin", "xmax", "ymin", "ymax"], axis=1
).columns

statistic_analisis = data_total_enrich[cols_to_select].describe()

with pd.ExcelWriter("analisis_descriptivo_etiquetas.xlsx") as writer:
    data_total_enrich.to_excel(writer, sheet_name="base enriquecida")
    statistic_analisis.to_excel(writer, sheet_name="estadistica descriptiva")


# jupyter nbconvert --to script understand_data.ipynb --output understanding_labels.py
