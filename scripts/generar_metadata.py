import csv
from pathlib import Path

base_dir = Path("datasets/yolo/images")
labels_dir = Path("datasets/original/labels")
out_csv = Path("datasets/original/metadata.csv")

CLASES_SOLDADURA = {3, 5} 

rows = []
for split in ["train", "val", "test"]:
    folder = base_dir / split
    if not folder.exists():
        continue
    for img in sorted(folder.iterdir()):
        if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            lbl_path = labels_dir / f"{img.stem}.txt"
            actividad = "corte"
            if lbl_path.exists():
                clases_en_foto = {
                    int(line.split()[0])
                    for line in lbl_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                }
                if clases_en_foto & CLASES_SOLDADURA:
                    actividad = "soldadura"

            grupo = img.stem.split("_")[0]
            rows.append([img.name, actividad, grupo, "internet", split])

with out_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["archivo", "actividad", "grupo_sesion", "fuente", "division"])
    writer.writerows(rows)

sold = sum(1 for r in rows if r[1] == "soldadura")
print(f"Listo: {len(rows)} filas ({sold} soldadura | {len(rows) - sold} corte)")