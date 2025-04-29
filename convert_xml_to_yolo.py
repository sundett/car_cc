import os
import xml.etree.ElementTree as ET

def convert(xml_folder, output_folder, classes):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(xml_folder):
        if not filename.endswith(".xml"):
            continue

        tree = ET.parse(os.path.join(xml_folder, filename))
        root = tree.getroot()

        image_width = int(root.find('size/width').text)
        image_height = int(root.find('size/height').text)

        output_lines = []

        for obj in root.findall('object'):
            cls = obj.find('name').text
            if cls not in classes:
                print(f"Класс '{cls}' аннотацияда табылмады!")  # Егер класс табылмаса
                continue
            cls_id = classes.index(cls)

            xml_box = obj.find('bndbox')
            xmin = int(float(xml_box.find('xmin').text))
            ymin = int(float(xml_box.find('ymin').text))
            xmax = int(float(xml_box.find('xmax').text))
            ymax = int(float(xml_box.find('ymax').text))

            # YOLO формат: координаттарды 0-1 аралығында беру
            x_center = (xmin + xmax) / 2.0 / image_width
            y_center = (ymin + ymax) / 2.0 / image_height
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height

            output_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        if output_lines:
            output_txt = os.path.join(output_folder, filename.replace(".xml", ".txt"))
            with open(output_txt, "w") as f:
                f.write("\n".join(output_lines))
        else:
            print(f"Аннотация үшін файл бос: {filename}")

    print("✅ Конвертация аяқталды.")

# Пайдалану:
# xml файлдар қайда орналасқанын көрсет
xml_folder = r'C:\Users\Nursultan\Desktop\1111\filters\project\dataset\annotations\val'  # XML папкасы
output_folder = r'C:\Users\Nursultan\Desktop\1111\filters\project\dataset\labels\val'  # YOLO форматында сақталатын папка
classes = ['license_plate']  # Жаңа класс атауы (YOLO үшін)

convert(xml_folder, output_folder, classes)
