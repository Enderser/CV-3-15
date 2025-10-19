import sys
import os
import cv2
import numpy as np
import argparse
from mtcnn import MTCNN
from typing import List, Tuple, Optional


class FacePortraitCropper:
    """
    Класс для обнаружения лиц и создания портретов с расширенной областью
    """
    def __init__(self, expand_percentage: float = 0.2):
        """
        Args:
            expand_percentage (float): Процент расширения bounding box (по умолчанию 20%)
        """
        self.detector = MTCNN()
        self.expand_percentage = expand_percentage
    
    def read_and_preprocess(self, img_array: np.ndarray) -> np.ndarray:
        """
        Предобработка изображения для детекции лиц
        
        Args:
            img_array (np.ndarray): Входное изображение
            
        Returns:
            np.ndarray: Предобработанное изображение
        """
        if img_array.dtype != np.uint8:
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)
                
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            
        return img_array
    
    def expand_bbox(self, x: int, y: int, w: int, h: int, 
                   img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """
        Расширяет bounding box на заданный процент
        
        Args:
            x, y, w, h: Координаты и размеры bounding box
            img_width, img_height: Размеры исходного изображения
            
        Returns:
            Tuple: Новые координаты (x, y, w, h)
        """
        expand_w = int(w * self.expand_percentage)
        expand_h = int(h * self.expand_percentage)
        
        new_x = max(0, x - expand_w // 2)
        new_y = max(0, y - expand_h // 2)
        new_w = min(img_width - new_x, w + expand_w)
        new_h = min(img_height - new_y, h + expand_h)
        
        return new_x, new_y, new_w, new_h
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Обнаруживает лица на изображении
        
        Args:
            image (np.ndarray): Входное изображение
            
        Returns:
            List: Список bounding boxes в формате (x, y, w, h)
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = self.detector.detect_faces(image_rgb)
        
        faces = []
        for detection in detections:
            if detection['confidence'] > 0.9:
                x, y, w, h = detection['box']
                faces.append((x, y, w, h))
                
        return faces
    
    def crop_portrait(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Обрезает изображение по расширенному bounding box
        
        Args:
            image (np.ndarray): Исходное изображение
            bbox (Tuple): Bounding box (x, y, w, h)
            
        Returns:
            np.ndarray: Обрезанное изображение-портрет
        """
        x, y, w, h = bbox
        height, width = image.shape[:2]
        
        expanded_bbox = self.expand_bbox(x, y, w, h, width, height)
        ex, ey, ew, eh = expanded_bbox
        
        portrait = image[ey:ey+eh, ex:ex+ew]
        
        return portrait
    
    def process_single_image(self, input_path: str, output_path: str) -> List[str]:
        """
        Обрабатывает одно изображение: находит лица и сохраняет портреты
        
        Args:
            input_path (str): Путь к входному изображению
            output_path (str): Путь для сохранения результата
            
        Returns:
            List[str]: Список путей к сохраненным портретам
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Could not load image: {input_path}")
        
        image = self.read_and_preprocess(image)
        
        faces = self.detect_faces(image)
        
        saved_paths = []
        
        for i, face in enumerate(faces):
            portrait = self.crop_portrait(image, face)
            
            base_name = os.path.splitext(output_path)[0]
            ext = os.path.splitext(output_path)[1]
            
            if len(faces) > 1:
                portrait_path = f"{base_name}_face_{i+1}{ext}"
            else:
                portrait_path = output_path
            
            os.makedirs(os.path.dirname(portrait_path) if os.path.dirname(portrait_path) else '.', 
                       exist_ok=True)
            
            success = cv2.imwrite(portrait_path, portrait)
            if success:
                saved_paths.append(portrait_path)
                print(f"Saved portrait: {portrait_path} (Face {i+1})")
            else:
                print(f"Failed to save: {portrait_path}")
        
        return saved_paths
    
    def process_image_series(self, input_dir: str, output_dir: str) -> List[str]:
        """
        Обрабатывает серию изображений в директории
        
        Args:
            input_dir (str): Директория с входными изображениями
            output_dir (str): Директория для сохранения портретов
            
        Returns:
            List[str]: Список всех сохраненных портретов
        """
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        all_saved_paths = []
        
        for filename in os.listdir(input_dir):
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in supported_formats:
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"portrait_{filename}")
                
                try:
                    saved_paths = self.process_single_image(input_path, output_path)
                    all_saved_paths.extend(saved_paths)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        return all_saved_paths


def main():
    parser = argparse.ArgumentParser(description="Face detection and portrait cropping system")
    parser.add_argument("-i", "--input", required=True, 
                       help="Input image file or directory")
    parser.add_argument("-o", "--output", required=True,
                       help="Output image file or directory")
    parser.add_argument("--expand", type=float, default=0.2,
                       help="Bounding box expansion percentage (default: 0.2 for 20%)")
    parser.add_argument("--batch", action="store_true",
                       help="Process multiple images in batch mode")
    
    args = parser.parse_args()
    
    cropper = FacePortraitCropper(expand_percentage=args.expand)
    
    try:
        if args.batch:
            print(f"Processing image series from: {args.input}")
            saved_paths = cropper.process_image_series(args.input, args.output)
            print(f"Successfully processed {len(saved_paths)} portraits")
        else:
            print(f"Processing single image: {args.input}")
            saved_paths = cropper.process_single_image(args.input, args.output)
            print(f"Successfully created {len(saved_paths)} portrait(s)")
            
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()