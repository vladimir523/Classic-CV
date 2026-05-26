import os
from pathlib import Path
os.environ["QT_LOGGING_RULES"] = "*.warning=false"
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image


info = {
    "1": ["short_arc", "long_arc", "long_arc", 1, 2, 1],
    "2": ["short_arc", "straight", "short_arc", 1, 1, 1],
    "3": ["short_arc", "short_arc", "short_arc", 1, 1, 1],
    "4": ["long_arc", "straight", "long_arc", 2, 2, 1],
    "5": ["short_arc", "short_arc", "straight", 1, 1, 1],
    "6": ["straight", "long_arc", "long_arc", 2, 1, 2],
    "7": ["long_arc", "short_arc", "long_arc", 2, 1, 1],
    "8": ["long_arc", "short_arc", "long_arc", 1, 1, 2],
    "9": ["long_arc", "long_arc", "straight", 2, 1, 2],
    "10": ["short_arc", "long_arc", "long_arc", 1, 1, 2],
}


help_img_path = "info_numbers.png"


def read(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Не удалось открыть изображение")
    return img


def dark(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 0, 0])
    upper = np.array([180, 255, 175])

    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = np.zeros_like(mask)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            cv2.drawContours(result, [contour], -1, 255, -1)

    return result


def group_mask(img):
    mask = dark(img)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)

    res = np.zeros_like(mask)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 1000:
            continue

        comp = np.zeros_like(mask)
        comp[labels == i] = 255

        num_1, _ = cv2.connectedComponents(comp, connectivity=4)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        comp = cv2.erode(comp, kernel, iterations=1)

        num_2, _ = cv2.connectedComponents(comp, connectivity=4)

        if num_1 == num_2:
            comp = cv2.dilate(comp, kernel, iterations=1)

        res = cv2.bitwise_or(res, comp)

    contours, _ = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = np.zeros_like(res)

    for c in contours:
        area = cv2.contourArea(c)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)

        if hull_area == 0:
            continue

        solidity = area / hull_area

        if solidity > 0.9:
            cv2.drawContours(result, [c], -1, 255, -1)

    return result

def count(img):
    mask = group_mask(img)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    good = []
    for c in contours:
        if cv2.contourArea(c) > 1000:
            good.append(c)

    return len(good), mask, good

def one_tile(img):
    mask = dark(img)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area > best_area:
            best_area = area
            best = c

    if best is None:
        return None, None, None

    x, y, w, h = cv2.boundingRect(best)
    tile = img[y:y+h, x:x+w].copy()
    tile_mask = mask[y:y+h, x:x+w].copy()

    return tile, tile_mask, (x, y)


def color_mask(tile, tile_mask, color):
    hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)

    if color == "yellow":
        mask = cv2.inRange(hsv, np.array([16, 175, 100]), np.array([24, 255, 255]))
    elif color == "blue":
        mask = cv2.inRange(hsv, np.array([100, 0, 0]), np.array([178, 255, 255]))
    elif color == "red":
        mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([8, 255, 255]))
    else:
        return None

    mask = cv2.bitwise_and(mask, tile_mask)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num, labels = cv2.connectedComponents(mask)

    for i in range(1, num):
        comp = labels == i
        if np.count_nonzero(comp) < 50:
            mask[comp] = 0

    return mask

def line_types(y_mask, b_mask, r_mask, sides):
    res = {}
    masks = {
        "yellow": y_mask,
        "blue": b_mask,
        "red": r_mask
    }

    for color, mask in masks.items():
        ys, xs = np.where(mask > 0)

        if len(xs) < 50:
            res[color] = "not_found"
            continue

        pts = np.column_stack((xs, ys))
        d = []

        for i, (cx, cy) in enumerate(sides):
            dx = pts[:, 0] - cx
            dy = pts[:, 1] - cy
            dist = dx * dx + dy * dy
            d.append((np.min(dist), i))

        d.sort()

        s1 = d[0][1]
        s2 = d[1][1]

        diff = abs(s1 - s2)
        diff = min(diff, 6 - diff)

        if diff == 1:
            t = "short_arc"
        elif diff == 2:
            t = "long_arc"
        elif diff == 3:
            t = "straight"
        else:
            t = "unknown"

        res[color] = t

    return res


def points(tile_mask):
    m = cv2.moments(tile_mask)

    if m["m00"] == 0:
        return None, [], []

    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])
    center = (cx, cy)

    contours, _ = cv2.findContours(tile_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return center, [], []

    contour = max(contours, key=cv2.contourArea)
    pts = contour.reshape(-1, 2)

    best = None
    best_dist = -1

    for x, y in pts:
        dist = (x - cx) ** 2 + (y - cy) ** 2
        if dist > best_dist:
            best_dist = dist
            best = (int(x), int(y))

    rad = best_dist ** 0.5
    vx = best[0] - cx
    vy = best[1] - cy

    approx = []

    for i in range(6):
        angle = i * np.pi / 3
        x = cx + vx * np.cos(angle) - vy * np.sin(angle)
        y = cy + vx * np.sin(angle) + vy * np.cos(angle)
        approx.append((int(x), int(y)))

    verts = []
    search_rad = rad / 3

    for ax, ay in approx:
        best_p = None
        best_d = -1

        for x, y in pts:
            d1 = ((x - ax) ** 2 + (y - ay) ** 2) ** 0.5

            if d1 < search_rad:
                d2 = (x - cx) ** 2 + (y - cy) ** 2

                if d2 > best_d:
                    best_d = d2
                    best_p = (int(x), int(y))

        if best_p is not None:
            verts.append(best_p)
        else:
            verts.append((ax, ay))

    sides = []

    for i in range(6):
        p1 = verts[i]
        p2 = verts[(i + 1) % 6]

        x = int((p1[0] + p2[0]) / 2)
        y = int((p1[1] + p2[1]) / 2)

        sides.append((x, y))

    return center, verts, sides


def parts(y_mask, b_mask, r_mask):
    res = {}
    masks = {
        "yellow": y_mask,
        "blue": b_mask,
        "red": r_mask
    }

    for color, mask in masks.items():
        num, _ = cv2.connectedComponents(mask)
        res[color] = num - 1

    return res


def tile_num(types, comps):
    cur = [
        types["yellow"],
        types["blue"],
        types["red"],
        comps["yellow"],
        comps["blue"],
        comps["red"]
    ]

    for k, v in info.items():
        if v == cur:
            return int(k)

    for k, v in info.items():
        if v[0] == cur[0] and v[2] == cur[2] and v[3] == cur[3] and v[5] == cur[5]:
            return int(k)

    return "?"

def rec(tile, tile_mask):
    _, _, sides = points(tile_mask)

    y_mask = color_mask(tile, tile_mask, "yellow")
    b_mask = color_mask(tile, tile_mask, "blue")
    r_mask = color_mask(tile, tile_mask, "red")

    types = line_types(y_mask, b_mask, r_mask, sides)
    comps = parts(y_mask, b_mask, r_mask)

    return tile_num(types, comps), types, comps


def cv_to_pil(img):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def fit_image(img, max_w, max_h):
    pil = cv_to_pil(img)
    w, h = pil.size
    k = min(max_w / w, max_h / h, 1)

    if k < 1:
        pil = pil.resize((int(w * k), int(h * k)))

    return pil


def ru_type(t):
    if t == "short_arc":
        return "короткая"
    if t == "long_arc":
        return "длинная"
    if t == "straight":
        return "прямая"
    return t


def ru_color(c):
    if c == "yellow":
        return "желтая"
    if c == "blue":
        return "синяя"
    if c == "red":
        return "красная"
    return c


def ru_color_word(c):
    if c == "yellow":
        return "желтая"
    if c == "blue":
        return "синяя"
    if c == "red":
        return "красная"
    return c


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Tantrix Recognizer")
        self.geometry("1180x720")
        self.minsize(1000, 620)

        self.img = None
        self.path = ""
        self.job = ctk.IntVar(value=1)
        self.main_image = None
        self.last_image = None
        self.help_open = False

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(9, weight=1)

        self.title_label = ctk.CTkLabel(
            self.sidebar,
            text="Tantrix",
            font=ctk.CTkFont(size=30, weight="bold")
        )
        self.title_label.grid(row=0, column=0, padx=22, pady=(24, 4), sticky="w")

        self.subtitle = ctk.CTkLabel(
            self.sidebar,
            text="Распознавание фишек",
            text_color="#9ca3af"
        )
        self.subtitle.grid(row=1, column=0, padx=22, pady=(0, 20), sticky="w")

        self.path_box = ctk.CTkTextbox(self.sidebar, height=58, width=250)
        self.path_box.grid(row=2, column=0, padx=22, pady=(0, 12), sticky="ew")
        self.path_box.insert("1.0", "Файл не выбран")
        self.path_box.configure(state="disabled")

        self.open_btn = ctk.CTkButton(
            self.sidebar,
            text="Выбрать изображение",
            height=40,
            command=self.open_file
        )
        self.open_btn.grid(row=3, column=0, padx=22, pady=(0, 16), sticky="ew")

        self.job_label = ctk.CTkLabel(
            self.sidebar,
            text="Задание",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.job_label.grid(row=4, column=0, padx=22, pady=(4, 8), sticky="w")

        self.jobs_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.jobs_frame.grid(row=5, column=0, padx=16, pady=0, sticky="ew")

        names = [
            "1. Количество фишек",
            "2. Типы линий",
            "3. Номер фишки",
            "4. Все фишки"
        ]

        for i, name in enumerate(names, start=1):
            rb = ctk.CTkRadioButton(
                self.jobs_frame,
                text=name,
                variable=self.job,
                value=i
            )
            rb.pack(anchor="w", pady=5)

        self.run_btn = ctk.CTkButton(
            self.sidebar,
            text="Запустить",
            height=44,
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self.run
        )
        self.run_btn.grid(row=6, column=0, padx=22, pady=(18, 8), sticky="ew")

        self.help_btn = ctk.CTkButton(
            self.sidebar,
            text="Справочник фишек",
            height=36,
            fg_color="#2563eb",
            hover_color="#1d4ed8",
            command=self.show_help
        )
        self.help_btn.grid(row=7, column=0, padx=22, pady=(0, 8), sticky="ew")

        self.clear_btn = ctk.CTkButton(
            self.sidebar,
            text="Очистить вывод",
            height=36,
            fg_color="#374151",
            hover_color="#4b5563",
            command=self.clear
        )
        self.clear_btn.grid(row=8, column=0, padx=22, pady=(0, 12), sticky="ew")

        self.log_box = ctk.CTkTextbox(self.sidebar, width=250)
        self.log_box.grid(row=9, column=0, padx=22, pady=(0, 22), sticky="nsew")
        self.log_box.insert("1.0", "Выберите изображение и задание.\n")
        self.log_box.configure(state="disabled")
        self.log_box._textbox.tag_config("yellow", foreground="#facc15")
        self.log_box._textbox.tag_config("blue", foreground="#60a5fa")
        self.log_box._textbox.tag_config("red", foreground="#f87171")

        self.content = ctk.CTkFrame(self, corner_radius=18)
        self.content.grid(row=0, column=1, padx=18, pady=18, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(1, weight=1)

        self.header = ctk.CTkFrame(self.content, fg_color="transparent")
        self.header.grid(row=0, column=0, sticky="ew", padx=20, pady=(18, 8))
        self.header.grid_columnconfigure(0, weight=1)

        self.image_title = ctk.CTkLabel(
            self.header,
            text="Просмотр",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.image_title.grid(row=0, column=0, sticky="w")

        self.status = ctk.CTkLabel(
            self.header,
            text="Готово",
            text_color="#9ca3af"
        )
        self.status.grid(row=0, column=1, sticky="e")

        self.image_frame = ctk.CTkFrame(self.content, fg_color="#111827", corner_radius=14)
        self.image_frame.grid(row=1, column=0, padx=20, pady=(8, 20), sticky="nsew")
        self.image_frame.grid_columnconfigure(0, weight=1)
        self.image_frame.grid_rowconfigure(0, weight=1)

        self.image_label = ctk.CTkLabel(self.image_frame, text="Изображение не выбрано", text_color="#6b7280")
        self.image_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

    def log(self, text):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", str(text) + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def log_color_line(self, line_type, color):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", line_type + " ")
        start = self.log_box.index("end-1c")
        self.log_box.insert("end", ru_color_word(color))
        end = self.log_box.index("end-1c")
        self.log_box._textbox.tag_add(color, start, end)
        self.log_box.insert("end", "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def clear(self):
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    def set_path_text(self, text):
        self.path_box.configure(state="normal")
        self.path_box.delete("1.0", "end")
        self.path_box.insert("1.0", text)
        self.path_box.configure(state="disabled")

    def show(self, img, save=True):
        if save:
            self.last_image = img.copy()

        pil = fit_image(img, 820, 560)
        self.main_image = ctk.CTkImage(light_image=pil, dark_image=pil, size=pil.size)
        self.image_label.configure(image=self.main_image, text="")

    def show_help(self):
        if self.help_open:
            self.help_open = False
            self.help_btn.configure(text="Справочник фишек")

            if self.last_image is not None:
                self.show(self.last_image, save=False)
                self.status.configure(text="Возврат к результату")
            elif self.img is not None:
                self.show(self.img, save=False)
                self.status.configure(text="Возврат к изображению")

            return

        try:
            img = read(help_img_path)
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            return

        self.help_open = True
        self.help_btn.configure(text="Вернуться назад")
        self.clear()
        self.log("Справочник фишек")
        self.status.configure(text="Справочник открыт")
        self.show(img, save=False)

    def open_file(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("Images", "*.bmp *.png *.jpg *.jpeg"),
                ("All files", "*.*")
            ]
        )

        if path == "":
            return

        try:
            self.path = path
            self.img = read(path)
            self.set_path_text(path)
            self.help_open = False
            self.help_btn.configure(text="Справочник фишек")
            self.show(self.img)
            self.status.configure(text="Файл загружен")
            self.clear()
            self.log("Файл загружен")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def need_image(self):
        if self.img is None:
            messagebox.showerror("Ошибка", "Сначала выберите изображение")
            return False
        return True

    def check_file(self, job):
        name = Path(self.path).name

        if job in [1, 4]:
            if not name.startswith("Group_"):
                messagebox.showerror("Ошибка", "Для задания 1 и 4 нужен файл Group_*.bmp")
                return False

        if job in [2, 3]:
            if not name.startswith("Single_"):
                messagebox.showerror("Ошибка", "Для задания 2 и 3 нужен файл Single_*.bmp")
                return False

        if not name.lower().endswith(".bmp"):
            messagebox.showerror("Ошибка", "Нужен файл формата BMP")
            return False

        return True

    def run(self):
        if not self.need_image():
            return

        self.help_open = False
        self.help_btn.configure(text="Справочник фишек")

        self.clear()
        job = self.job.get()

        if not self.check_file(job):
            return

        try:
            if job == 1:
                self.task1()
            elif job == 2:
                self.task2()
            elif job == 3:
                self.task3()
            elif job == 4:
                self.task4()
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def task1(self):
        n, mask, contours = count(self.img)

        res = self.img.copy()
        cv2.drawContours(res, contours, -1, (0, 255, 0), 2)

        self.log(f"Количество фишек: {n}")
        self.status.configure(text="Задание 1 выполнено")
        self.show(res)

    def task2(self):
        tile, tile_mask, _ = one_tile(self.img)

        if tile is None:
            self.log("Фишка не найдена")
            self.show(self.img)
            return

        _, _, sides = points(tile_mask)

        y_mask = color_mask(tile, tile_mask, "yellow")
        b_mask = color_mask(tile, tile_mask, "blue")
        r_mask = color_mask(tile, tile_mask, "red")

        types = line_types(y_mask, b_mask, r_mask, sides)

        self.log_color_line(ru_type(types['yellow']), "yellow")
        self.log_color_line(ru_type(types['blue']), "blue")
        self.log_color_line(ru_type(types['red']), "red")
        self.status.configure(text="Задание 2 выполнено")
        self.show(self.img)

    def task3(self):
        tile, tile_mask, _ = one_tile(self.img)

        if tile is None:
            self.log("Фишка не найдена")
            self.show(self.img)
            return

        n, _, _ = rec(tile, tile_mask)

        self.log(f"Номер фишки: {n}")
        self.status.configure(text="Задание 3 выполнено")
        self.show(self.img)

    def task4(self):
        mask = group_mask(self.img)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        res = self.img.copy()
        found = 0

        for c in contours:
            if cv2.contourArea(c) < 1000:
                continue

            x, y, w, h = cv2.boundingRect(c)

            tile = self.img[y:y+h, x:x+w]
            tile_mask = mask[y:y+h, x:x+w]

            n, _, _ = rec(tile, tile_mask)

            cx = x + w // 2
            cy = y + h // 2

            cv2.rectangle(res, (cx - 20, cy - 20), (cx + 20, cy + 20), (255, 255, 255), -1)
            cv2.putText(res, str(n), (cx - 10, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

            found += 1
            self.log(f"Фишка: {n}")

        self.log(f"Всего фишек: {found}")
        self.status.configure(text="Задание 4 выполнено")
        self.show(res)


if __name__ == "__main__":
    app = App()
    app.mainloop()