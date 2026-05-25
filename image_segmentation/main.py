import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import FastSAM
import matplotlib.pyplot as plt
import urllib.request
import threading


def preprocess_image(image, target_size=1024):
    # ШАГ 0: РЕСАЙЗ
    h, w = image.shape[:2]
    max_dim = max(h, w)
    
    if max_dim > target_size:
        scale = target_size / max_dim
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        img = image.copy()

    # ШАГ 1: РАЗМЫТИЕ  
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # ШАГ 2: ГАММА-КОРРЕКЦИЯ 
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    gamma = 0.6
    
    v_float = v.astype(np.float32) / 255.0
    v_corrected = np.power(v_float, gamma)
    v_enhanced = (v_corrected * 255).astype(np.uint8)
    
    hsv_lit = cv2.merge([h, s, v_enhanced])
    img_lit = cv2.cvtColor(hsv_lit, cv2.COLOR_HSV2BGR)

    # ШАГ 3: ПОВЫШЕНИЕ ЯРКОСТИ 
    beta = 10  
    
    final_image = cv2.convertScaleAbs(img_lit, beta=beta)

    return final_image

def segment_with_fastsam(image, model_path="FastSAM-x.pt", erosion_kernel_size=5):
    try:
        model = FastSAM(model_path)
        results = model(image, device='cpu', retina_masks=True, imgsz=1024, conf=0.45, iou=0.3)
    except Exception as e:
        print(f"Ошибка FastSAM: {e}")
        return []
    
    masks = []
    if results and len(results) > 0:
        for result in results:
            if result.masks is not None:
                result_masks = result.masks.data.cpu().numpy()
                masks.extend(result_masks)

    if not masks:
        return []

    binary_masks = [(mask > 0.5).astype(np.uint8) for mask in masks]
    areas = [m.sum() for m in binary_masks]
    
    # ШАГ 1: УДАЛЕНИЕ ВЫБРОСОВ ПО ПЛОЩАДИ
    masks_to_remove = set()
    
    for i, area in enumerate(areas):
            if area < 400:
                masks_to_remove.add(i) 

    if masks_to_remove:
        masks = [m for idx, m in enumerate(masks) if idx not in masks_to_remove]
        binary_masks = [m for idx, m in enumerate(binary_masks) if idx not in masks_to_remove]
        areas = [m.sum() for m in binary_masks] 
    
    masks_to_remove = set()
    
    if len(masks) > 3:
        median_area = np.median(areas)
        MIN_AREA_RATIO = 0.5 
        MAX_AREA_RATIO = 1.75  
        
        for i, area in enumerate(areas):
            if area < median_area * MIN_AREA_RATIO or area > median_area * MAX_AREA_RATIO:
                masks_to_remove.add(i)
    
    if masks_to_remove:
        masks = [m for idx, m in enumerate(masks) if idx not in masks_to_remove]
        binary_masks = [m for idx, m in enumerate(binary_masks) if idx not in masks_to_remove]
        areas = [m.sum() for m in binary_masks] 
    
    if not masks:
        return []

    # ШАГ 2: УДАЛЕНИЕ ВЛОЖЕННОСТЕЙ
    masks_to_remove = set()
    num_masks = len(masks)
    
    for i in range(num_masks):
        if i in masks_to_remove: continue
            
        for j in range(i + 1, num_masks):
            if j in masks_to_remove: continue
            
            intersection = np.logical_and(binary_masks[i], binary_masks[j]).sum()
            
            if intersection < 50: continue
            
            min_area = min(areas[i], areas[j])
            max_area = max(areas[i], areas[j])
            
            if min_area == 0: continue
            
            overlap_ratio = intersection / min_area
            area_ratio = min_area / max_area
            
            if overlap_ratio > 0.9:  # Почти полная вложенность
                if area_ratio < 0.4:  # Маленькая << Большой (Пятно)
                    masks_to_remove.add(i if areas[i] < areas[j] else j)
                else:               # Сопоставимые размеры 
                    masks_to_remove.add(i if areas[i] > areas[j] else j)
            
            elif overlap_ratio > 0.5: # Сильное пересечение (Дубликат)
                if areas[i] > areas[j] * 1.3:
                    masks_to_remove.add(i)
                elif areas[j] > areas[i] * 1.3:
                    masks_to_remove.add(j)
    
    if masks_to_remove:
        masks = [m for idx, m in enumerate(masks) if idx not in masks_to_remove]
        binary_masks = [m for idx, m in enumerate(binary_masks) if idx not in masks_to_remove]
    
    # ШАГ 3: ЭРОЗИЯ КОНТУРОВ
    if erosion_kernel_size > 1 and masks:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_kernel_size, erosion_kernel_size))
        
        eroded_masks = []
        for bin_mask in binary_masks: 
            eroded = cv2.erode(bin_mask, kernel, iterations=1)
            
            if eroded.sum() > 50:
                eroded_masks.append(eroded.astype(np.float32))
        
        masks = eroded_masks
    
    return masks

def classify_and_count(masks, original_image):
    red_tomatoes, yellow_tomatoes, eggs, other = 0, 0, 0, 0
    classifications = []
    
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    
    SATURATION_THRESHOLD = 151
    BRIGHTNESS_THRESHOLD = 140

    object_stats = []
    
    # ШАГ 1: Сбор статистики по маскам
    for mask in masks:
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
        
        if cv2.countNonZero(binary_mask) == 0:
            object_stats.append({'pixels': None})
            continue
            
        masked_hsv = cv2.bitwise_and(hsv_image, hsv_image, mask=binary_mask)
        object_pixels = masked_hsv[binary_mask > 0]
        
        if len(object_pixels) == 0:
            object_stats.append({'pixels': None})
            continue
        
        mean_hsv = np.mean(object_pixels, axis=0)
        std_hsv = np.std(object_pixels, axis=0)
        
        object_stats.append({
            'pixels': object_pixels,
            'mean_hsv': mean_hsv,   
            'std_hsv': std_hsv,     
            'texture_std': std_hsv.mean()
        })
    
    # ШАГ 2: Разделение на Помидоры и Яйца 
    tomato_indices = []
    
    for i, stats in enumerate(object_stats):
        if stats['pixels'] is None:
            continue
        
        if stats['mean_hsv'][1] >= SATURATION_THRESHOLD:
            tomato_indices.append(i)
            
    # ШАГ 3: Классификация цвета помидоров
    tomato_color_map = {}

    if tomato_indices:
        indexed_values = [(i, object_stats[i]['mean_hsv'][2]) for i in tomato_indices]
        
        MIN_GAP_THRESHOLD = 5.0  

        if len(indexed_values) >= 2:
            sorted_values = sorted(indexed_values, key=lambda x: x[1])
            
            max_gap = 0
            split_threshold = None
            
            for k in range(len(sorted_values) - 1):
                current_val = sorted_values[k][1]
                next_val = sorted_values[k+1][1]
                gap = next_val - current_val
                
                if gap > max_gap:
                    max_gap = gap
                    split_threshold = (current_val + next_val) / 2
            
            if split_threshold is not None and max_gap > MIN_GAP_THRESHOLD:
                for i, val in indexed_values:
                    tomato_color_map[i] = 'yellow_tomato' if val >= split_threshold else 'red_tomato'
            else:
                for i, val in indexed_values:
                    tomato_color_map[i] = 'yellow_tomato' if val >= BRIGHTNESS_THRESHOLD else 'red_tomato'
                    
        else:
            i = tomato_indices[0]
            val = object_stats[i]['mean_hsv'][2]
            tomato_color_map[i] = 'yellow_tomato' if val >= BRIGHTNESS_THRESHOLD else 'red_tomato'

    # ШАГ 4 Подсчет
    for i, stats in enumerate(object_stats):
        if stats['pixels'] is None:
            classifications.append('unrecognized')
            other += 1
            continue
        
        if i in tomato_indices:
            label = tomato_color_map.get(i, 'red_tomato')
            classifications.append(label)
            if label == 'red_tomato':
                red_tomatoes += 1
            else:
                yellow_tomatoes += 1
        else:
            classifications.append('egg')
            eggs += 1
    
    return {
        "counts": {
            "red_tomatoes": red_tomatoes, 
            "yellow_tomatoes": yellow_tomatoes, 
            "eggs": eggs, 
            "total": red_tomatoes + yellow_tomatoes + eggs + other
        },
        "classifications": classifications
    }

def visualize_classified_masks(original_image_shape, masks, classifications):
    mask_image = np.zeros(original_image_shape, dtype=np.uint8)

    COLOR_MAP = {
        'red_tomato':    (0, 0, 255),     
        'yellow_tomato': (0, 255, 255),    
        'egg':           (0, 255, 0),     
        'unrecognized':  (255, 0, 0)       
    }
    
    DEFAULT_COLOR = (255, 0, 0) 

    for i, mask in enumerate(masks):
        class_label = classifications[i] if i < len(classifications) else 'unrecognized'
        
        color = COLOR_MAP.get(class_label, DEFAULT_COLOR)

        binary_mask = (mask > 0.5).astype(np.uint8)

        mask_image[binary_mask > 0] = color
        
    return mask_image

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Система анализа изображений (Tomatoes & Eggs)")
        self.root.geometry("600x400")
        
        self.model_path = "FastSAM-x.pt"
        self.image_path = None
        self.original_image = None
        
        self.create_widgets()
        
        self.check_model()

    def create_widgets(self):
        title_label = tk.Label(self.root, text="Панель управления обработкой", font=("Arial", 16, "bold"))
        title_label.pack(pady=20)
        
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=20)
        
        self.btn_select = tk.Button(btn_frame, text="1. Выбрать изображение", command=self.select_file, width=25, height=2)
        self.btn_select.pack(pady=10)
        
        self.btn_process = tk.Button(btn_frame, text="2. Запустить обработку", command=self.start_processing, width=25, height=2, state=tk.DISABLED)
        self.btn_process.pack(pady=10)
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ожидание выбора файла...")
        self.status_label = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.file_info_label = tk.Label(self.root, text="", fg="blue", wraplength=500)
        self.file_info_label.pack(pady=10)

    def check_model(self):
        if not os.path.exists(self.model_path):
            self.status_var.set("Модель не найдена. Скачивание...")
            self.root.update()
            try:
                url = "https://github.com/ultralytics/assets/releases/download/v8.4.0/FastSAM-x.pt"
                urllib.request.urlretrieve(url, self.model_path)
                self.status_var.set(f"Модель успешно скачана: {self.model_path}")
            except Exception as e:
                self.status_var.set(f"Ошибка скачивания модели: {e}")
                messagebox.showerror("Ошибка", f"Не удалось скачать модель:\n{e}")
        else:
            self.status_var.set(f"Модель готова: {self.model_path}")

    def select_file(self):
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.image_path = file_path
            self.original_image = cv2.imread(file_path)
            
            if self.original_image is None:
                messagebox.showerror("Ошибка", "Не удалось прочитать изображение.")
                return
                
            filename = os.path.basename(file_path)
            self.file_info_label.config(text=f"Выбран файл: {filename}")
            self.btn_process.config(state=tk.NORMAL)
            self.status_var.set("Файл выбран. Готов к обработке.")

    def start_processing(self):
        if not self.image_path:
            return
            
        self.btn_select.config(state=tk.DISABLED)
        self.btn_process.config(state=tk.DISABLED)
        self.status_var.set("Обработка изображения... Пожалуйста, подождите.")
        self.root.update()
        
        thread = threading.Thread(target=self.run_processing_logic)
        thread.start()

    def run_processing_logic(self):
        try:
            self.status_var.set("Шаг 1: Предобработка (Resize, Blur, Gamma)...")
            self.root.update()
            preprocessed = preprocess_image(self.original_image)
            
            self.status_var.set("Шаг 2: Сегментация (FastSAM)...")
            self.root.update()
            masks = segment_with_fastsam(preprocessed, self.model_path)
            
            self.status_var.set("Шаг 3: Классификация и создание маски...")
            self.root.update()
            
            if not masks:
                final_image = self.original_image.copy()
                cv2.putText(final_image, "Objects not found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                counts = {"total": 0, "red_tomatoes": 0, "yellow_tomatoes": 0, "eggs": 0}
            else:
                results_data = classify_and_count(masks, self.original_image)
                counts = results_data["counts"]
                classifications = results_data["classifications"]
                
                mask_vis = visualize_classified_masks(self.original_image.shape, masks, classifications)
                final_image = cv2.addWeighted(self.original_image, 0.6, mask_vis, 0.4, 0)
                
                text = f"Total: {counts['total']} (R:{counts['red_tomatoes']}, Y:{counts['yellow_tomatoes']}, E:{counts['eggs']})"
                cv2.putText(final_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            self.status_var.set("Открытие окна с результатами...")
            self.root.update()
            
            original_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            preprocessed_rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
            final_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'Результаты: {os.path.basename(self.image_path)}', fontsize=16)
            
            axes[0].imshow(original_rgb)
            axes[0].set_title('1. Original')
            axes[0].axis('off')
            
            axes[1].imshow(preprocessed_rgb)
            axes[1].set_title('2. Preprocessed')
            axes[1].axis('off')
            
            res_title = f"3. Final (Total: {counts['total']})"
            axes[2].imshow(final_rgb)
            axes[2].set_title(res_title)
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.show()
            
            results_folder = "results"
            os.makedirs(results_folder, exist_ok=True)
            filename = os.path.basename(self.image_path)
            save_path = os.path.join(results_folder, f"final_{filename}")
            cv2.imwrite(save_path, final_image)
            
            self.status_var.set(f"Готово! Результат сохранен в {save_path}")
            messagebox.showinfo("Успех", f"Обработка завершена!\n\nРезультаты:\nВсего: {counts['total']}\nКрасные: {counts['red_tomatoes']}\nЖелтые: {counts['yellow_tomatoes']}\nЯйца: {counts['eggs']}\n\nФайл сохранен: {save_path}")
            
        except Exception as e:
            self.status_var.set("Произошла ошибка при обработке.")
            messagebox.showerror("Ошибка", f"Критическая ошибка:\n{str(e)}")
            print(f"Error details: {e}")
        
        finally:
            self.btn_select.config(state=tk.NORMAL)
            self.btn_process.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()