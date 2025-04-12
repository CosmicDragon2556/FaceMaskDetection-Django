import os
import xml.etree.ElementTree as ET
import pandas as pd
import cv2
import matplotlib.pyplot as plt

class DataLoader:
    @staticmethod
    def load_dataset(image_dir, annotation_dir):
        """Parse XML annotations and return DataFrame"""
        data = []
        for xml_file in os.listdir(annotation_dir):
            if xml_file.endswith('.xml'):
                xml_path = os.path.join(annotation_dir, xml_file)
                tree = ET.parse(xml_path)
                root = tree.getroot()

                filename = root.find('filename').text
                width = int(root.find('size/width').text)
                height = int(root.find('size/height').text)

                for obj in root.findall('object'):
                    data.append([
                        filename,
                        width,
                        height,
                        obj.find('name').text,
                        int(obj.find('bndbox/xmin').text),
                        int(obj.find('bndbox/ymin').text),
                        int(obj.find('bndbox/xmax').text),
                        int(obj.find('bndbox/ymax').text)
                    ])
        return pd.DataFrame(data, columns=[
            'filename', 'width', 'height', 
            'class_name', 'xmin', 'ymin', 'xmax', 'ymax'
        ])

    @staticmethod
    def prepare_cnn_dataset(df, image_dir, output_dir='dataset_cnn'):
        """Crop faces and organize into class folders"""
        os.makedirs(output_dir, exist_ok=True)
        
        for image_name in df['filename'].unique():
            image_path = os.path.join(image_dir, image_name)
            image = cv2.imread(image_path)
            
            for _, obj in df[df['filename'] == image_name].iterrows():
                face = image[obj['ymin']:obj['ymax'], obj['xmin']:obj['xmax']]
                class_dir = os.path.join(output_dir, obj['class_name'])
                os.makedirs(class_dir, exist_ok=True)
                cv2.imwrite(
                    os.path.join(class_dir, f"{image_name}_{obj['xmin']}_{obj['ymin']}.jpg"),
                    face
                )

    @staticmethod
    def visualize_samples(df, image_dir, num_samples=5):
        """Plot samples with bounding boxes"""
        plt.figure(figsize=(15, 10))
        for i in range(num_samples):
            sample = df.sample(1).iloc[0]
            img = cv2.cvtColor(cv2.imread(os.path.join(image_dir, sample['filename'])), cv2.COLOR_BGR2RGB)
            
            cv2.rectangle(img, (sample['xmin'], sample['ymin']), 
                         (sample['xmax'], sample['ymax']), (0,255,0), 2)
            cv2.putText(img, sample['class_name'], (sample['xmin'], sample['ymin']-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            
            plt.subplot(1, num_samples, i+1)
            plt.imshow(img)
            plt.axis('off')
        plt.show()