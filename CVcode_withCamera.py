import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import os

class CoinNutScrewDetectorGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("硬币、螺母及螺钉检测器")
        self.master.geometry("1800x1000")  # 调整窗口大小以适应新的布局

        # 默认参数
        self.parameters = {
            'dp': 1.2,
            'minDist': 97,
            'param1': 50,
            'param2': 30,
            'minRadius_coin': 36,
            'maxRadius_coin': 63,
            'canny_threshold1': 50,
            'canny_threshold2': 150,
            'brightness_threshold': 200,  # 亮度阈值，用于检测反光区域
            'screw_aspect_ratio_min': 1.4,  # 螺钉长宽比下限
            'screw_aspect_ratio_max': 10.0,  # 螺钉长宽比上限
            'nut_min_area': 140,  # 螺母最小面积
            'nut_max_area': 5000,  # 螺母最大面积
            'nut_circularity': 0.3,  # 螺母圆形度阈值
            'screw_min_area': 800,  # 螺钉最小面积
            'screw_max_area': 10000,   # 螺钉最大面积
            'resize_width': 800,        # 图像缩放宽度
            'resize_height': 600        # 图像缩放高度
        }

        # 初始化检测锁和防抖定时器
        self.detection_lock = threading.Lock()
        self.debounce_timer = None

        # 实时识别相关变量
        self.video_thread = None
        self.video_running = False
        self.cap = None

        # 设置GUI布局
        self.setup_ui()

        self.image_path = "input.jpg"
        self.original_image = None  # 存储原始图像
        self.detected_image = None  # 存储检测结果图像
        if os.path.exists(self.image_path):
            self.load_image(self.image_path)
            self.start_detection_thread()
        else:
            messagebox.showwarning("警告", f"默认图像 '{self.image_path}' 未找到。请使用“加载图片”按钮选择图像。")

    def setup_ui(self):
        # 创建主Frame
        main_frame = tk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 上方：检测结果信息
        result_info_frame = tk.Frame(main_frame)
        result_info_frame.pack(side=tk.TOP, fill=tk.X, pady=10)

        self.coin_count_label = tk.Label(result_info_frame, text="检测到的硬币数量: 0", font=("Arial", 14))
        self.coin_count_label.pack(side=tk.LEFT, padx=20)

        self.nut_count_label = tk.Label(result_info_frame, text="检测到的螺母数量: 0", font=("Arial", 14))
        self.nut_count_label.pack(side=tk.LEFT, padx=20)

        self.screw_count_label = tk.Label(result_info_frame, text="检测到的螺钉数量: 0", font=("Arial", 14))
        self.screw_count_label.pack(side=tk.LEFT, padx=20)

        # 中间：检测结果图像和参数滑块
        middle_frame = tk.Frame(main_frame)
        middle_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧：参数滑块区域，包含滚动条
        sliders_frame_container = tk.Frame(middle_frame)
        sliders_frame_container.pack(side=tk.LEFT, fill=tk.Y)

        # 创建Canvas和滚动条
        self.sliders_canvas = tk.Canvas(sliders_frame_container, width=400)
        self.sliders_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar_sliders = tk.Scrollbar(sliders_frame_container, orient=tk.VERTICAL, command=self.sliders_canvas.yview)
        self.scrollbar_sliders.pack(side=tk.RIGHT, fill=tk.Y)

        self.sliders_canvas.configure(yscrollcommand=self.scrollbar_sliders.set)

        # 创建一个Frame在Canvas中，用于放置滑块
        self.sliders_container = tk.Frame(self.sliders_canvas)
        self.sliders_canvas.create_window((0, 0), window=self.sliders_container, anchor='nw')

        # 绑定Canvas大小变化以更新scrollregion
        self.sliders_container.bind("<Configure>", lambda e: self.sliders_canvas.configure(scrollregion=self.sliders_canvas.bbox("all")))

        # 绑定鼠标滚轮事件
        self.bind_mousewheel(self.sliders_canvas, "sliders")

        # 创建左右两列滑块的Frame
        sliders_left_frame = tk.Frame(self.sliders_container)
        sliders_left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=5)

        sliders_right_frame = tk.Frame(self.sliders_container)
        sliders_right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=5)

        # 添加滑块到左半部分
        self.sliders = {}
        left_slider_names = [
            ("dp", 1.0, 3.0, 0.1, self.parameters['dp']),
            ("minDist", 10, 200, 1, self.parameters['minDist']),
            ("param1", 10, 200, 1, self.parameters['param1']),
            ("param2", 10, 100, 1, self.parameters['param2']),
            ("minRadius_coin", 0, 100, 1, self.parameters['minRadius_coin']),
            ("maxRadius_coin", 0, 200, 1, self.parameters['maxRadius_coin']),
            ("canny_threshold1", 0, 300, 1, self.parameters['canny_threshold1']),
            ("canny_threshold2", 0, 300, 1, self.parameters['canny_threshold2'])
        ]

        for name, frm, to, res, default in left_slider_names:
            self.create_slider(sliders_left_frame, name, frm, to, res, default)

        # 添加滑块到右半部分
        right_slider_names = [
            ("brightness_threshold", 0, 255, 1, self.parameters['brightness_threshold']),
            ("nut_min_area", 50, 1000, 10, self.parameters['nut_min_area']),
            ("nut_max_area", 500, 10000, 50, self.parameters['nut_max_area']),
            ("nut_circularity", 0.1, 1.0, 0.05, self.parameters['nut_circularity']),
            ("screw_aspect_ratio_min", 1.0, 5.0, 0.1, self.parameters['screw_aspect_ratio_min']),
            ("screw_aspect_ratio_max", 5.0, 15.0, 0.1, self.parameters['screw_aspect_ratio_max']),
            ("screw_min_area", 100, 20000, 100, self.parameters['screw_min_area']),
            ("screw_max_area", 5000, 30000, 500, self.parameters['screw_max_area']),
            ("resize_width", 200, 1600, 50, self.parameters['resize_width']),
            ("resize_height", 200, 1200, 50, self.parameters['resize_height'])
        ]

        for name, frm, to, res, default in right_slider_names:
            self.create_slider(sliders_right_frame, name, frm, to, res, default)

        # 右侧：图像显示区域，包含滚动条
        images_frame_container = tk.Frame(middle_frame)
        images_frame_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 创建Canvas和滚动条
        self.images_canvas = tk.Canvas(images_frame_container)
        self.images_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar_images = tk.Scrollbar(images_frame_container, orient=tk.VERTICAL, command=self.images_canvas.yview)
        self.scrollbar_images.pack(side=tk.RIGHT, fill=tk.Y)

        self.images_canvas.configure(yscrollcommand=self.scrollbar_images.set)

        # 创建一个Frame在Canvas中，用于放置图像标签
        self.images_container = tk.Frame(self.images_canvas)
        self.images_canvas.create_window((0, 0), window=self.images_container, anchor='nw')

        # 绑定Canvas大小变化以更新scrollregion
        self.images_container.bind("<Configure>", lambda e: self.images_canvas.configure(scrollregion=self.images_canvas.bbox("all")))

        # 绑定鼠标滚轮事件
        self.bind_mousewheel(self.images_canvas, "images")

        # 创建图像显示的Frame
        images_frame = tk.Frame(self.images_container)
        images_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 检测结果图
        self.result_label = tk.Label(images_frame, text="检测结果", font=("Arial", 14))
        self.result_label.pack(pady=5)
        self.result_canvas = tk.Label(images_frame)
        self.result_canvas.pack(pady=5)

        # 边缘检测图
        self.edge_label = tk.Label(images_frame, text="边缘检测图", font=("Arial", 14))
        self.edge_label.pack(pady=5)
        self.edge_canvas = tk.Label(images_frame)
        self.edge_canvas.pack(pady=5)

        # 显示使用的边缘检测算子
        self.edge_detectors_label = tk.Label(images_frame, text="使用的边缘检测算子: Canny, Sobel (X和Y方向), Laplacian", font=("Arial", 12), fg="blue")
        self.edge_detectors_label.pack(pady=5)

        # 实时识别图像显示区域
        self.real_time_label = tk.Label(images_frame, text="实时识别结果", font=("Arial", 14))
        self.real_time_label.pack(pady=5)
        self.real_time_canvas = tk.Label(images_frame)
        self.real_time_canvas.pack(pady=5)

        # 控制按钮框架（位于参数滑块下方）
        control_buttons_frame = tk.Frame(main_frame)
        control_buttons_frame.pack(side=tk.BOTTOM, pady=10)

        # 加载图片按钮
        load_button = tk.Button(control_buttons_frame, text="加载图片", command=self.browse_image, width=15, height=2)
        load_button.pack(side=tk.LEFT, padx=20)

        # 保存结果按钮
        save_button = tk.Button(control_buttons_frame, text="保存结果", command=self.save_result, width=15, height=2)
        save_button.pack(side=tk.LEFT, padx=20)

        # 实时识别按钮
        self.realtime_button = tk.Button(control_buttons_frame, text="实时识别", command=self.toggle_realtime, width=15, height=2, bg="green")
        self.realtime_button.pack(side=tk.LEFT, padx=20)

        # 停止实时识别按钮（初始隐藏）
        self.stop_realtime_button = tk.Button(control_buttons_frame, text="停止识别", command=self.stop_realtime, width=15, height=2, bg="red")
        # self.stop_realtime_button.pack(side=tk.LEFT, padx=20)  # 仅在实时识别启动后显示

    def create_slider(self, parent, name, from_, to, resolution, default):
        frame = tk.Frame(parent)
        frame.pack(pady=5, fill=tk.X)

        label = tk.Label(frame, text=name, font=("Arial", 12))
        label.pack(anchor='w')

        if isinstance(default, float):
            slider = tk.Scale(
                frame, from_=from_, to=to, orient=tk.HORIZONTAL,
                resolution=resolution, length=200,
                command=self.on_parameter_change
            )
        else:
            slider = tk.Scale(
                frame, from_=from_, to=to, orient=tk.HORIZONTAL,
                resolution=resolution, length=200,
                command=self.on_parameter_change
            )
        slider.set(default)
        slider.pack(fill=tk.X)
        self.sliders[name] = slider

    def on_parameter_change(self, event=None):
        """参数改变时触发检测，使用防抖机制"""
        if self.debounce_timer:
            self.master.after_cancel(self.debounce_timer)
        self.debounce_timer = self.master.after(500, self.start_detection_thread)

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff")]
        )
        if file_path:
            self.image_path = file_path
            self.load_image(self.image_path)
            self.start_detection_thread()

    def load_image(self, path):
        print(f"加载图像: {path}")
        # 读取图像
        image = cv2.imread(path)
        if image is None:
            messagebox.showerror("错误", f"无法读取图像文件: {path}")
            return

        # 获取缩放参数
        resize_width = self.sliders['resize_width'].get()
        resize_height = self.sliders['resize_height'].get()

        # 缩放图像
        image_resized = cv2.resize(image, (resize_width, resize_height))
        self.original_image = image_resized  # 存储缩放后的图像

        # 转换为灰度图
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        # 应用高斯滤波
        gray_filtered = cv2.GaussianBlur(gray, (9, 9), 2)

        # 融合边缘检测
        edges = self.fused_edge_detection(gray_filtered)

        # 显示边缘检测图
        self.display_image(edges, self.edge_canvas, 'edges')

    def fused_edge_detection(self, gray_image):
        """
        使用多种边缘检测算子进行融合边缘检测。
        算子包括：Canny, Sobel (X和Y方向), Laplacian。
        """
        # Canny 边缘检测
        canny_threshold1 = self.sliders['canny_threshold1'].get()
        canny_threshold2 = self.sliders['canny_threshold2'].get()
        edges_canny = cv2.Canny(gray_image, canny_threshold1, canny_threshold2)
        print("应用 Canny 边缘检测。")

        # Sobel 边缘检测（X方向）
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = cv2.convertScaleAbs(sobelx)
        print("应用 Sobel X 边缘检测。")

        # Sobel 边缘检测（Y方向）
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        sobely = cv2.convertScaleAbs(sobely)
        print("应用 Sobel Y 边缘检测。")

        # Laplacian 边缘检测
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        laplacian = cv2.convertScaleAbs(laplacian)
        print("应用 Laplacian 边缘检测。")

        # 融合边缘检测结果
        edges_fused = cv2.bitwise_or(edges_canny, sobelx)
        edges_fused = cv2.bitwise_or(edges_fused, sobely)
        edges_fused = cv2.bitwise_or(edges_fused, laplacian)
        print("融合边缘检测结果。")

        return edges_fused

    def display_image(self, img, canvas, img_type):
        """在指定的Canvas上显示图像"""
        try:
            if len(img.shape) == 2:  # 灰度图
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            # 获取当前缩放尺寸
            resize_width = self.sliders['resize_width'].get()
            resize_height = self.sliders['resize_height'].get()
            img_pil = img_pil.resize((resize_width, resize_height), Image.LANCZOS)

            # 创建PhotoImage
            photo = ImageTk.PhotoImage(img_pil)

            # 更新Canvas上的图像
            canvas.config(image=photo)
            canvas.image = photo  # 保持引用

            print(f"显示图像: {img_type}, 尺寸: {img_pil.size}")
        except Exception as e:
            print(f"显示图像时出错 ({img_type}): {e}")
            messagebox.showerror("错误", f"显示图像时出错 ({img_type}): {e}")

    def start_detection_thread(self):
        """启动一个新线程进行检测，以保持GUI响应"""
        if self.detection_lock.locked():
            print("检测线程已在运行，跳过新的检测请求。")
            return  # 避免同时运行多个检测线程
        detection_thread = threading.Thread(target=self.detect_objects)
        detection_thread.daemon = True  # 设置为守护线程
        detection_thread.start()
        print("启动检测线程。")

    def detect_objects(self):
        with self.detection_lock:
            try:
                print("开始检测硬币、螺母及螺钉...")
                # 获取当前参数
                dp = self.sliders['dp'].get()
                minDist = self.sliders['minDist'].get()
                param1 = self.sliders['param1'].get()
                param2 = self.sliders['param2'].get()
                minRadius_coin = self.sliders['minRadius_coin'].get()
                maxRadius_coin = self.sliders['maxRadius_coin'].get()
                canny_threshold1 = self.sliders['canny_threshold1'].get()
                canny_threshold2 = self.sliders['canny_threshold2'].get()
                brightness_threshold = self.sliders['brightness_threshold'].get()
                screw_aspect_ratio_min = self.sliders['screw_aspect_ratio_min'].get()
                screw_aspect_ratio_max = self.sliders['screw_aspect_ratio_max'].get()
                nut_min_area = self.sliders['nut_min_area'].get()
                nut_max_area = self.sliders['nut_max_area'].get()
                nut_circularity = self.sliders['nut_circularity'].get()
                screw_min_area = self.sliders['screw_min_area'].get()
                screw_max_area = self.sliders['screw_max_area'].get()
                resize_width = self.sliders['resize_width'].get()
                resize_height = self.sliders['resize_height'].get()

                print(f"当前参数: dp={dp}, minDist={minDist}, param1={param1}, param2={param2}, "
                      f"minRadius_coin={minRadius_coin}, maxRadius_coin={maxRadius_coin}, "
                      f"canny_threshold1={canny_threshold1}, canny_threshold2={canny_threshold2}, "
                      f"brightness_threshold={brightness_threshold}, "
                      f"screw_aspect_ratio_min={screw_aspect_ratio_min}, "
                      f"screw_aspect_ratio_max={screw_aspect_ratio_max}, "
                      f"nut_min_area={nut_min_area}, nut_max_area={nut_max_area}, "
                      f"nut_circularity={nut_circularity}, screw_min_area={screw_min_area}, "
                      f"screw_max_area={screw_max_area}, resize_width={resize_width}, resize_height={resize_height}")

                if self.original_image is None:
                    self.show_error("尚未加载任何图像。")
                    return

                image = self.original_image.copy()
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # 处理光泽反光：检测亮区域并应用模糊
                bright_mask = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)[1]
                bright_mask = cv2.GaussianBlur(bright_mask, (15, 15), 0)
                image_blurred = cv2.GaussianBlur(image, (15, 15), 0)
                image = np.where(bright_mask[:, :, np.newaxis] == 255, image_blurred, image)

                # 转换为灰度图
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # 应用高斯滤波
                gray_filtered = cv2.GaussianBlur(gray, (9, 9), 2)

                # 融合边缘检测
                edges = self.fused_edge_detection(gray_filtered)

                # 显示边缘检测图
                self.master.after(0, self.display_image, edges, self.edge_canvas, 'edges')

                # 应用形态学闭运算以闭合边缘中的断裂
                kernel = np.ones((3, 3), np.uint8)
                closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

                # 初始化检测计数和检测结果列表
                detections = []
                output = image.copy()

                # 检测硬币
                circles = cv2.HoughCircles(
                    closed_edges,
                    cv2.HOUGH_GRADIENT,
                    dp=dp,
                    minDist=minDist,
                    param1=param1,
                    param2=param2,
                    minRadius=minRadius_coin,
                    maxRadius=maxRadius_coin
                )

                if circles is not None:
                    circles = np.round(circles[0, :]).astype("int")
                    print(f"检测到圆的数量（硬币候选）: {len(circles)}")

                    for (x, y, r) in circles:
                        area = np.pi * (r ** 2)
                        circularity = 1.0  # 硬币假设为完美圆形
                        if area >= 1000:  # 根据实际情况调整面积阈值
                            # 计算概率评分
                            prob_coin = 1.0  # 硬币的概率评分可以基于圆形度
                            detections.append({
                                'type': 'Coin',
                                'bbox': (x - r, y - r, x + r, y + r),
                                'score': prob_coin,
                                'center': (x, y),
                                'radius': r
                            })
                            print(f"硬币候选: 位置=({x}, {y}), 半径={r}, 面积={int(area)}")
                else:
                    print("未检测到任何硬币。")

                # 检测螺母和螺钉
                # 使用轮廓检测，因为螺母和螺钉不是完全圆形
                contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter == 0:
                        continue
                    circularity = 4 * np.pi * (area / (perimeter ** 2))
                    x, y, w, h = cv2.boundingRect(cnt)

                    # 计算中心
                    cx = x + w // 2
                    cy = y + h // 2

                    # 螺母检测
                    if nut_min_area <= area <= nut_max_area and circularity >= nut_circularity:
                        prob_nut = circularity  # 螺母的概率评分基于圆形度
                        detections.append({
                            'type': 'Nut',
                            'bbox': (x, y, x + w, y + h),
                            'score': prob_nut,
                            'center': (cx, cy),
                            'bbox_size': (w, h)
                        })
                        print(f"螺母候选: 边界矩形=({x}, {y}, {w}, {h}), 面积={int(area)}, 圆形度={circularity:.2f}")

                    # 螺钉检测
                    if screw_min_area <= area <= screw_max_area:
                        # 使用最小外接旋转矩形
                        rect = cv2.minAreaRect(cnt)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        width = rect[1][0]
                        height = rect[1][1]
                        if width == 0 or height == 0:
                            continue
                        aspect_ratio = max(width, height) / min(width, height)
                        if screw_aspect_ratio_min <= aspect_ratio <= screw_aspect_ratio_max:
                            prob_screw = 1 / aspect_ratio  # 螺钉的概率评分基于长宽比，长宽比越小概率越高
                            # 计算边界框的四个坐标
                            x_min = min(box[:, 0])
                            y_min = min(box[:, 1])
                            x_max = max(box[:, 0])
                            y_max = max(box[:, 1])
                            detections.append({
                                'type': 'Screw',
                                'bbox': (x_min, y_min, x_max, y_max),  # 修正为四个值
                                'score': prob_screw,
                                'center': (int(rect[0][0]), int(rect[0][1])),
                                'box': box
                            })
                            print(f"螺钉候选: 旋转矩形=({box.tolist()}), 面积={int(area)}, 长宽比={aspect_ratio:.2f}")

                # 处理重叠检测，确保每个位置只有一种类型
                final_detections = self.resolve_overlaps(detections)

                # 重置计数
                coin_count = 0
                nut_count = 0
                screw_count = 0

                # 绘制最终检测结果
                for det in final_detections:
                    if det['type'] == 'Coin':
                        x1, y1, x2, y2 = det['bbox']
                        r = det.get('radius', (x2 - x1) // 2)
                        x_center = det['center'][0]
                        y_center = det['center'][1]
                        cv2.circle(output, (x_center, y_center), r, (0, 255, 0), 2)
                        cv2.circle(output, (x_center, y_center), 2, (0, 0, 255), 3)
                        cv2.putText(output, f"Coin {coin_count + 1}", (x_center - r, y_center - r - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        coin_count += 1
                    elif det['type'] == 'Nut':
                        x1, y1, x2, y2 = det['bbox']
                        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(output, f"Nut {nut_count + 1}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        nut_count += 1
                    elif det['type'] == 'Screw':
                        box = det['box']
                        cv2.drawContours(output, [box], 0, (0, 0, 255), 2)
                        cv2.putText(output, f"Screw {screw_count + 1}", tuple(box[0]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        screw_count += 1

                # 保存检测到的结果图像以供保存功能使用
                self.detected_image = output.copy()

                # 准备显示结果图像
                # 使用self.master.after确保在主线程中更新GUI
                self.master.after(0, self.update_gui, output, coin_count, nut_count, screw_count)
                print("检测完成，更新GUI。")
            except Exception as e:
                print(f"检测过程中出错: {e}")
                self.show_error(f"检测过程中出错: {e}")

    def resolve_overlaps(self, detections):
        """解析检测结果，消除重叠，只保留概率最高的检测"""
        # 按照得分从高到低排序
        detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        final_detections = []

        for det in detections:
            overlap = False
            for final_det in final_detections:
                if self.bboxes_overlap(det['bbox'], final_det['bbox']):
                    overlap = True
                    break
            if not overlap:
                final_detections.append(det)

        return final_detections

    def bboxes_overlap(self, bbox1, bbox2):
        """判断两个边界框是否重叠"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # 计算交集
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
            # 计算交集面积
            inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
            # 计算最小的边界框面积
            area1 = (x1_max - x1_min) * (y1_max - y1_min)
            area2 = (x2_max - x2_min) * (y2_max - y2_min)
            min_area = min(area1, area2)
            # 如果交集面积超过一定比例，则认为重叠
            if inter_area / min_area > 0.3:  # 阈值可调整
                return True
        return False

    def update_gui(self, output_image, coin_count, nut_count, screw_count):
        """在主线程中更新GUI"""
        try:
            self.display_image(output_image, self.result_canvas, 'result')
            self.update_result_info(coin_count, nut_count, screw_count)
            print("GUI 更新成功。")
        except Exception as e:
            print(f"更新GUI时出错: {e}")
            messagebox.showerror("错误", f"更新GUI时出错: {e}")

    def update_result_info(self, coin, nut, screw):
        """更新GUI中的检测结果信息"""
        self.coin_count_label.config(text=f"检测到的硬币数量: {coin}")
        self.nut_count_label.config(text=f"检测到的螺母数量: {nut}")
        self.screw_count_label.config(text=f"检测到的螺钉数量: {screw}")

    def show_error(self, message):
        """在主线程中显示错误消息"""
        self.master.after(0, lambda: messagebox.showerror("错误", message))

    def save_result(self):
        """保存检测结果图像"""
        if self.detected_image is not None:
            save_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), ("All files", "*.*")]
            )
            if save_path:
                try:
                    # 将OpenCV图像转换为RGB并保存
                    img_rgb = cv2.cvtColor(self.detected_image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(img_rgb)
                    pil_image.save(save_path)
                    messagebox.showinfo("保存成功", f"检测结果已保存到: {save_path}")
                    print(f"检测结果已保存到: {save_path}")
                except Exception as e:
                    print(f"保存图像时出错: {e}")
                    messagebox.showerror("错误", f"保存图像时出错: {e}")
        else:
            messagebox.showwarning("警告", "尚未进行检测或未检测到任何结果。")

    def bind_mousewheel(self, widget, target):
        """绑定鼠标滚轮事件到指定的widget"""
        # 针对不同操作系统绑定不同的事件
        if os.name == 'nt':  # Windows
            widget.bind("<Enter>", lambda e: widget.bind_all("<MouseWheel>", self.on_mousewheel_windows))
            widget.bind("<Leave>", lambda e: widget.unbind_all("<MouseWheel>"))
        elif os.name == 'posix':  # Linux
            widget.bind("<Enter>", lambda e: widget.bind_all("<Button-4>", self.on_mousewheel_linux))
            widget.bind("<Enter>", lambda e: widget.bind_all("<Button-5>", self.on_mousewheel_linux))
            widget.bind("<Leave>", lambda e: widget.unbind_all("<Button-4>"))
            widget.bind("<Leave>", lambda e: widget.unbind_all("<Button-5>"))
        else:  # macOS
            widget.bind("<Enter>", lambda e: widget.bind_all("<MouseWheel>", self.on_mousewheel_macos))
            widget.bind("<Leave>", lambda e: widget.unbind_all("<MouseWheel>"))

    def on_mousewheel_windows(self, event):
        """处理Windows上的鼠标滚轮事件"""
        if self.is_mouse_over_widget(self.sliders_canvas):
            self.sliders_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        elif self.is_mouse_over_widget(self.images_canvas):
            self.images_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def on_mousewheel_linux(self, event):
        """处理Linux上的鼠标滚轮事件"""
        if event.num == 4:
            if self.is_mouse_over_widget(self.sliders_canvas):
                self.sliders_canvas.yview_scroll(-1, "units")
            elif self.is_mouse_over_widget(self.images_canvas):
                self.images_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            if self.is_mouse_over_widget(self.sliders_canvas):
                self.sliders_canvas.yview_scroll(1, "units")
            elif self.is_mouse_over_widget(self.images_canvas):
                self.images_canvas.yview_scroll(1, "units")

    def on_mousewheel_macos(self, event):
        """处理macOS上的鼠标滚轮事件"""
        if self.is_mouse_over_widget(self.sliders_canvas):
            self.sliders_canvas.yview_scroll(int(-1*(event.delta)), "units")
        elif self.is_mouse_over_widget(self.images_canvas):
            self.images_canvas.yview_scroll(int(-1*(event.delta)), "units")

    def is_mouse_over_widget(self, widget):
        """检查鼠标是否位于指定widget上"""
        x, y = self.master.winfo_pointerxy()
        widget_x = widget.winfo_rootx()
        widget_y = widget.winfo_rooty()
        widget_width = widget.winfo_width()
        widget_height = widget.winfo_height()
        return widget_x <= x <= widget_x + widget_width and widget_y <= y <= widget_y + widget_height

    def toggle_realtime(self):
        """切换实时识别的启动和停止"""
        if not self.video_running:
            self.start_realtime()
        else:
            self.stop_realtime()

    def start_realtime(self):
        """启动实时识别"""
        self.cap = cv2.VideoCapture(1)  # 打开默认摄像头
        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开摄像头。")
            return

        self.video_running = True
        self.realtime_button.config(text="停止识别", bg="red")
        self.stop_realtime_button.pack(side=tk.LEFT, padx=20)  # 显示停止按钮

        self.video_thread = threading.Thread(target=self.realtime_detection)
        self.video_thread.daemon = True
        self.video_thread.start()
        print("实时识别已启动。")

    def stop_realtime(self):
        """停止实时识别"""
        self.video_running = False
        self.realtime_button.config(text="实时识别", bg="green")
        self.stop_realtime_button.pack_forget()  # 隐藏停止按钮
        if self.cap:
            self.cap.release()
            self.cap = None
        print("实时识别已停止。")

    def realtime_detection(self):
        """实时识别视频流中的对象"""
        while self.video_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取摄像头帧。")
                break

            # 获取当前参数
            dp = self.sliders['dp'].get()
            minDist = self.sliders['minDist'].get()
            param1 = self.sliders['param1'].get()
            param2 = self.sliders['param2'].get()
            minRadius_coin = self.sliders['minRadius_coin'].get()
            maxRadius_coin = self.sliders['maxRadius_coin'].get()
            canny_threshold1 = self.sliders['canny_threshold1'].get()
            canny_threshold2 = self.sliders['canny_threshold2'].get()
            brightness_threshold = self.sliders['brightness_threshold'].get()
            screw_aspect_ratio_min = self.sliders['screw_aspect_ratio_min'].get()
            screw_aspect_ratio_max = self.sliders['screw_aspect_ratio_max'].get()
            nut_min_area = self.sliders['nut_min_area'].get()
            nut_max_area = self.sliders['nut_max_area'].get()
            nut_circularity = self.sliders['nut_circularity'].get()
            screw_min_area = self.sliders['screw_min_area'].get()
            screw_max_area = self.sliders['screw_max_area'].get()
            resize_width = self.sliders['resize_width'].get()
            resize_height = self.sliders['resize_height'].get()

            # 缩放帧
            frame_resized = cv2.resize(frame, (resize_width, resize_height))
            image = frame_resized.copy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 处理光泽反光：检测亮区域并应用模糊
            bright_mask = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)[1]
            bright_mask = cv2.GaussianBlur(bright_mask, (15, 15), 0)
            image_blurred = cv2.GaussianBlur(image, (15, 15), 0)
            image = np.where(bright_mask[:, :, np.newaxis] == 255, image_blurred, image)

            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 应用高斯滤波
            gray_filtered = cv2.GaussianBlur(gray, (9, 9), 2)

            # 融合边缘检测
            edges = self.fused_edge_detection(gray_filtered)

            # 显示边缘检测图
            self.master.after(0, self.display_image, edges, self.edge_canvas, 'edges')

            # 应用形态学闭运算以闭合边缘中的断裂
            kernel = np.ones((3, 3), np.uint8)
            closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

            # 初始化检测计数和检测结果列表
            detections = []
            output = image.copy()

            # 检测硬币
            circles = cv2.HoughCircles(
                closed_edges,
                cv2.HOUGH_GRADIENT,
                dp=dp,
                minDist=minDist,
                param1=param1,
                param2=param2,
                minRadius=minRadius_coin,
                maxRadius=maxRadius_coin
            )

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    area = np.pi * (r ** 2)
                    circularity = 1.0  # 硬币假设为完美圆形
                    if area >= 1000:
                        prob_coin = 1.0
                        detections.append({
                            'type': 'Coin',
                            'bbox': (x - r, y - r, x + r, y + r),
                            'score': prob_coin,
                            'center': (x, y),
                            'radius': r
                        })

            # 检测螺母和螺钉
            contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * (area / (perimeter ** 2))
                x, y, w, h = cv2.boundingRect(cnt)

                # 计算中心
                cx = x + w // 2
                cy = y + h // 2

                # 螺母检测
                if nut_min_area <= area <= nut_max_area and circularity >= nut_circularity:
                    prob_nut = circularity
                    detections.append({
                        'type': 'Nut',
                        'bbox': (x, y, x + w, y + h),
                        'score': prob_nut,
                        'center': (cx, cy),
                        'bbox_size': (w, h)
                    })

                # 螺钉检测
                if screw_min_area <= area <= screw_max_area:
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    width = rect[1][0]
                    height = rect[1][1]
                    if width == 0 or height == 0:
                        continue
                    aspect_ratio = max(width, height) / min(width, height)
                    if screw_aspect_ratio_min <= aspect_ratio <= screw_aspect_ratio_max:
                        prob_screw = 1 / aspect_ratio
                        x_min = min(box[:, 0])
                        y_min = min(box[:, 1])
                        x_max = max(box[:, 0])
                        y_max = max(box[:, 1])
                        detections.append({
                            'type': 'Screw',
                            'bbox': (x_min, y_min, x_max, y_max),
                            'score': prob_screw,
                            'center': (int(rect[0][0]), int(rect[0][1])),
                            'box': box
                        })

            # 处理重叠检测，确保每个位置只有一种类型
            final_detections = self.resolve_overlaps(detections)

            # 重置计数
            coin_count = 0
            nut_count = 0
            screw_count = 0

            # 绘制最终检测结果
            for det in final_detections:
                if det['type'] == 'Coin':
                    x1, y1, x2, y2 = det['bbox']
                    r = det.get('radius', (x2 - x1) // 2)
                    x_center = det['center'][0]
                    y_center = det['center'][1]
                    cv2.circle(output, (x_center, y_center), r, (0, 255, 0), 2)
                    cv2.circle(output, (x_center, y_center), 2, (0, 0, 255), 3)
                    cv2.putText(output, f"Coin {coin_count + 1}", (x_center - r, y_center - r - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    coin_count += 1
                elif det['type'] == 'Nut':
                    x1, y1, x2, y2 = det['bbox']
                    cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(output, f"Nut {nut_count + 1}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    nut_count += 1
                elif det['type'] == 'Screw':
                    box = det['box']
                    cv2.drawContours(output, [box], 0, (0, 0, 255), 2)
                    cv2.putText(output, f"Screw {screw_count + 1}", tuple(box[0]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    screw_count += 1

            # 保存检测到的结果图像以供保存功能使用
            self.detected_image = output.copy()

            # 准备显示结果图像
            # 使用self.master.after确保在主线程中更新GUI
            self.master.after(0, self.update_gui, output, coin_count, nut_count, screw_count)
            print("检测完成，更新GUI。")

    def resolve_overlaps(self, detections):
        """解析检测结果，消除重叠，只保留概率最高的检测"""
        # 按照得分从高到低排序
        detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        final_detections = []

        for det in detections:
            overlap = False
            for final_det in final_detections:
                if self.bboxes_overlap(det['bbox'], final_det['bbox']):
                    overlap = True
                    break
            if not overlap:
                final_detections.append(det)

        return final_detections

    def bboxes_overlap(self, bbox1, bbox2):
        """判断两个边界框是否重叠"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # 计算交集
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
            # 计算交集面积
            inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
            # 计算最小的边界框面积
            area1 = (x1_max - x1_min) * (y1_max - y1_min)
            area2 = (x2_max - x2_min) * (y2_max - y2_min)
            min_area = min(area1, area2)
            # 如果交集面积超过一定比例，则认为重叠
            if inter_area / min_area > 0.3:  # 阈值可调整
                return True
        return False

    def update_gui(self, output_image, coin_count, nut_count, screw_count):
        """在主线程中更新GUI"""
        try:
            self.display_image(output_image, self.result_canvas, 'result')
            self.update_result_info(coin_count, nut_count, screw_count)
            print("GUI 更新成功。")
        except Exception as e:
            print(f"更新GUI时出错: {e}")
            messagebox.showerror("错误", f"更新GUI时出错: {e}")

    def update_result_info(self, coin, nut, screw):
        """更新GUI中的检测结果信息"""
        self.coin_count_label.config(text=f"检测到的硬币数量: {coin}")
        self.nut_count_label.config(text=f"检测到的螺母数量: {nut}")
        self.screw_count_label.config(text=f"检测到的螺钉数量: {screw}")

    def show_error(self, message):
        """在主线程中显示错误消息"""
        self.master.after(0, lambda: messagebox.showerror("错误", message))

    def save_result(self):
        """保存检测结果图像"""
        if self.detected_image is not None:
            save_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), ("All files", "*.*")]
            )
            if save_path:
                try:
                    # 将OpenCV图像转换为RGB并保存
                    img_rgb = cv2.cvtColor(self.detected_image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(img_rgb)
                    pil_image.save(save_path)
                    messagebox.showinfo("保存成功", f"检测结果已保存到: {save_path}")
                    print(f"检测结果已保存到: {save_path}")
                except Exception as e:
                    print(f"保存图像时出错: {e}")
                    messagebox.showerror("错误", f"保存图像时出错: {e}")
        else:
            messagebox.showwarning("警告", "尚未进行检测或未检测到任何结果。")

    def bind_mousewheel(self, widget, target):
        """绑定鼠标滚轮事件到指定的widget"""
        # 针对不同操作系统绑定不同的事件
        if os.name == 'nt':  # Windows
            widget.bind("<Enter>", lambda e: widget.bind_all("<MouseWheel>", self.on_mousewheel_windows))
            widget.bind("<Leave>", lambda e: widget.unbind_all("<MouseWheel>"))
        elif os.name == 'posix':  # Linux
            widget.bind("<Enter>", lambda e: widget.bind_all("<Button-4>", self.on_mousewheel_linux))
            widget.bind("<Enter>", lambda e: widget.bind_all("<Button-5>", self.on_mousewheel_linux))
            widget.bind("<Leave>", lambda e: widget.unbind_all("<Button-4>"))
            widget.bind("<Leave>", lambda e: widget.unbind_all("<Button-5>"))
        else:  # macOS
            widget.bind("<Enter>", lambda e: widget.bind_all("<MouseWheel>", self.on_mousewheel_macos))
            widget.bind("<Leave>", lambda e: widget.unbind_all("<MouseWheel>"))

    def on_mousewheel_windows(self, event):
        """处理Windows上的鼠标滚轮事件"""
        if self.is_mouse_over_widget(self.sliders_canvas):
            self.sliders_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        elif self.is_mouse_over_widget(self.images_canvas):
            self.images_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def on_mousewheel_linux(self, event):
        """处理Linux上的鼠标滚轮事件"""
        if event.num == 4:
            if self.is_mouse_over_widget(self.sliders_canvas):
                self.sliders_canvas.yview_scroll(-1, "units")
            elif self.is_mouse_over_widget(self.images_canvas):
                self.images_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            if self.is_mouse_over_widget(self.sliders_canvas):
                self.sliders_canvas.yview_scroll(1, "units")
            elif self.is_mouse_over_widget(self.images_canvas):
                self.images_canvas.yview_scroll(1, "units")

    def on_mousewheel_macos(self, event):
        """处理macOS上的鼠标滚轮事件"""
        if self.is_mouse_over_widget(self.sliders_canvas):
            self.sliders_canvas.yview_scroll(int(-1*(event.delta)), "units")
        elif self.is_mouse_over_widget(self.images_canvas):
            self.images_canvas.yview_scroll(int(-1*(event.delta)), "units")

    def is_mouse_over_widget(self, widget):
        """检查鼠标是否位于指定widget上"""
        x, y = self.master.winfo_pointerxy()
        widget_x = widget.winfo_rootx()
        widget_y = widget.winfo_rooty()
        widget_width = widget.winfo_width()
        widget_height = widget.winfo_height()
        return widget_x <= x <= widget_x + widget_width and widget_y <= y <= widget_y + widget_height

    def toggle_realtime(self):
        """切换实时识别的启动和停止"""
        if not self.video_running:
            self.start_realtime()
        else:
            self.stop_realtime()

    def start_realtime(self):
        """启动实时识别"""
        self.cap = cv2.VideoCapture(1)  # 打开默认摄像头
        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开摄像头。")
            return

        self.video_running = True
        self.realtime_button.config(text="停止识别", bg="red")
        self.stop_realtime_button.pack(side=tk.LEFT, padx=20)  # 显示停止按钮

        self.video_thread = threading.Thread(target=self.realtime_detection)
        self.video_thread.daemon = True
        self.video_thread.start()
        print("实时识别已启动。")

    def stop_realtime(self):
        """停止实时识别"""
        self.video_running = False
        self.realtime_button.config(text="实时识别", bg="green")
        self.stop_realtime_button.pack_forget()  # 隐藏停止按钮
        if self.cap:
            self.cap.release()
            self.cap = None
        print("实时识别已停止。")

    def realtime_detection(self):
        """实时识别视频流中的对象"""
        while self.video_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取摄像头帧。")
                break

            # 获取当前参数
            dp = self.sliders['dp'].get()
            minDist = self.sliders['minDist'].get()
            param1 = self.sliders['param1'].get()
            param2 = self.sliders['param2'].get()
            minRadius_coin = self.sliders['minRadius_coin'].get()
            maxRadius_coin = self.sliders['maxRadius_coin'].get()
            canny_threshold1 = self.sliders['canny_threshold1'].get()
            canny_threshold2 = self.sliders['canny_threshold2'].get()
            brightness_threshold = self.sliders['brightness_threshold'].get()
            screw_aspect_ratio_min = self.sliders['screw_aspect_ratio_min'].get()
            screw_aspect_ratio_max = self.sliders['screw_aspect_ratio_max'].get()
            nut_min_area = self.sliders['nut_min_area'].get()
            nut_max_area = self.sliders['nut_max_area'].get()
            nut_circularity = self.sliders['nut_circularity'].get()
            screw_min_area = self.sliders['screw_min_area'].get()
            screw_max_area = self.sliders['screw_max_area'].get()
            resize_width = self.sliders['resize_width'].get()
            resize_height = self.sliders['resize_height'].get()

            # 缩放帧
            frame_resized = cv2.resize(frame, (resize_width, resize_height))
            image = frame_resized.copy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 处理光泽反光：检测亮区域并应用模糊
            bright_mask = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)[1]
            bright_mask = cv2.GaussianBlur(bright_mask, (15, 15), 0)
            image_blurred = cv2.GaussianBlur(image, (15, 15), 0)
            image = np.where(bright_mask[:, :, np.newaxis] == 255, image_blurred, image)

            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 应用高斯滤波
            gray_filtered = cv2.GaussianBlur(gray, (9, 9), 2)

            # 融合边缘检测
            edges = self.fused_edge_detection(gray_filtered)

            # 显示边缘检测图
            self.master.after(0, self.display_image, edges, self.edge_canvas, 'edges')

            # 应用形态学闭运算以闭合边缘中的断裂
            kernel = np.ones((3, 3), np.uint8)
            closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

            # 初始化检测计数和检测结果列表
            detections = []
            output = image.copy()

            # 检测硬币
            circles = cv2.HoughCircles(
                closed_edges,
                cv2.HOUGH_GRADIENT,
                dp=dp,
                minDist=minDist,
                param1=param1,
                param2=param2,
                minRadius=minRadius_coin,
                maxRadius=maxRadius_coin
            )

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    area = np.pi * (r ** 2)
                    circularity = 1.0  # 硬币假设为完美圆形
                    if area >= 1000:
                        prob_coin = 1.0
                        detections.append({
                            'type': 'Coin',
                            'bbox': (x - r, y - r, x + r, y + r),
                            'score': prob_coin,
                            'center': (x, y),
                            'radius': r
                        })

            # 检测螺母和螺钉
            contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * (area / (perimeter ** 2))
                x, y, w, h = cv2.boundingRect(cnt)

                # 计算中心
                cx = x + w // 2
                cy = y + h // 2

                # 螺母检测
                if nut_min_area <= area <= nut_max_area and circularity >= nut_circularity:
                    prob_nut = circularity
                    detections.append({
                        'type': 'Nut',
                        'bbox': (x, y, x + w, y + h),
                        'score': prob_nut,
                        'center': (cx, cy),
                        'bbox_size': (w, h)
                    })

                # 螺钉检测
                if screw_min_area <= area <= screw_max_area:
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    width = rect[1][0]
                    height = rect[1][1]
                    if width == 0 or height == 0:
                        continue
                    aspect_ratio = max(width, height) / min(width, height)
                    if screw_aspect_ratio_min <= aspect_ratio <= screw_aspect_ratio_max:
                        prob_screw = 1 / aspect_ratio
                        x_min = min(box[:, 0])
                        y_min = min(box[:, 1])
                        x_max = max(box[:, 0])
                        y_max = max(box[:, 1])
                        detections.append({
                            'type': 'Screw',
                            'bbox': (x_min, y_min, x_max, y_max),
                            'score': prob_screw,
                            'center': (int(rect[0][0]), int(rect[0][1])),
                            'box': box
                        })

            # 处理重叠检测，确保每个位置只有一种类型
            final_detections = self.resolve_overlaps(detections)

            # 重置计数
            coin_count = 0
            nut_count = 0
            screw_count = 0

            # 绘制最终检测结果
            for det in final_detections:
                if det['type'] == 'Coin':
                    x1, y1, x2, y2 = det['bbox']
                    r = det.get('radius', (x2 - x1) // 2)
                    x_center = det['center'][0]
                    y_center = det['center'][1]
                    cv2.circle(output, (x_center, y_center), r, (0, 255, 0), 2)
                    cv2.circle(output, (x_center, y_center), 2, (0, 0, 255), 3)
                    cv2.putText(output, f"Coin {coin_count + 1}", (x_center - r, y_center - r - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    coin_count += 1
                elif det['type'] == 'Nut':
                    x1, y1, x2, y2 = det['bbox']
                    cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(output, f"Nut {nut_count + 1}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    nut_count += 1
                elif det['type'] == 'Screw':
                    box = det['box']
                    cv2.drawContours(output, [box], 0, (0, 0, 255), 2)
                    cv2.putText(output, f"Screw {screw_count + 1}", tuple(box[0]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    screw_count += 1

            # 保存检测到的结果图像以供保存功能使用
            self.detected_image = output.copy()

            # 准备显示结果图像
            # 使用self.master.after确保在主线程中更新GUI
            self.master.after(0, self.update_gui, output, coin_count, nut_count, screw_count)
            print("检测完成，更新GUI。")

    def update_gui(self, output_image, coin_count, nut_count, screw_count):
        """在主线程中更新GUI"""
        try:
            self.display_image(output_image, self.result_canvas, 'result')
            self.update_result_info(coin_count, nut_count, screw_count)
            print("GUI 更新成功。")
        except Exception as e:
            print(f"更新GUI时出错: {e}")
            messagebox.showerror("错误", f"更新GUI时出错: {e}")

    def update_result_info(self, coin, nut, screw):
        """更新GUI中的检测结果信息"""
        self.coin_count_label.config(text=f"检测到的硬币数量: {coin}")
        self.nut_count_label.config(text=f"检测到的螺母数量: {nut}")
        self.screw_count_label.config(text=f"检测到的螺钉数量: {screw}")

    def show_error(self, message):
        """在主线程中显示错误消息"""
        self.master.after(0, lambda: messagebox.showerror("错误", message))

    def save_result(self):
        """保存检测结果图像"""
        if self.detected_image is not None:
            save_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), ("All files", "*.*")]
            )
            if save_path:
                try:
                    # 将OpenCV图像转换为RGB并保存
                    img_rgb = cv2.cvtColor(self.detected_image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(img_rgb)
                    pil_image.save(save_path)
                    messagebox.showinfo("保存成功", f"检测结果已保存到: {save_path}")
                    print(f"检测结果已保存到: {save_path}")
                except Exception as e:
                    print(f"保存图像时出错: {e}")
                    messagebox.showerror("错误", f"保存图像时出错: {e}")
        else:
            messagebox.showwarning("警告", "尚未进行检测或未检测到任何结果。")

    def bind_mousewheel(self, widget, target):
        """绑定鼠标滚轮事件到指定的widget"""
        # 针对不同操作系统绑定不同的事件
        if os.name == 'nt':  # Windows
            widget.bind("<Enter>", lambda e: widget.bind_all("<MouseWheel>", self.on_mousewheel_windows))
            widget.bind("<Leave>", lambda e: widget.unbind_all("<MouseWheel>"))
        elif os.name == 'posix':  # Linux
            widget.bind("<Enter>", lambda e: widget.bind_all("<Button-4>", self.on_mousewheel_linux))
            widget.bind("<Enter>", lambda e: widget.bind_all("<Button-5>", self.on_mousewheel_linux))
            widget.bind("<Leave>", lambda e: widget.unbind_all("<Button-4>"))
            widget.bind("<Leave>", lambda e: widget.unbind_all("<Button-5>"))
        else:  # macOS
            widget.bind("<Enter>", lambda e: widget.bind_all("<MouseWheel>", self.on_mousewheel_macos))
            widget.bind("<Leave>", lambda e: widget.unbind_all("<MouseWheel>"))

    def on_mousewheel_windows(self, event):
        """处理Windows上的鼠标滚轮事件"""
        if self.is_mouse_over_widget(self.sliders_canvas):
            self.sliders_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        elif self.is_mouse_over_widget(self.images_canvas):
            self.images_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def on_mousewheel_linux(self, event):
        """处理Linux上的鼠标滚轮事件"""
        if event.num == 4:
            if self.is_mouse_over_widget(self.sliders_canvas):
                self.sliders_canvas.yview_scroll(-1, "units")
            elif self.is_mouse_over_widget(self.images_canvas):
                self.images_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            if self.is_mouse_over_widget(self.sliders_canvas):
                self.sliders_canvas.yview_scroll(1, "units")
            elif self.is_mouse_over_widget(self.images_canvas):
                self.images_canvas.yview_scroll(1, "units")

    def on_mousewheel_macos(self, event):
        """处理macOS上的鼠标滚轮事件"""
        if self.is_mouse_over_widget(self.sliders_canvas):
            self.sliders_canvas.yview_scroll(int(-1*(event.delta)), "units")
        elif self.is_mouse_over_widget(self.images_canvas):
            self.images_canvas.yview_scroll(int(-1*(event.delta)), "units")

    def is_mouse_over_widget(self, widget):
        """检查鼠标是否位于指定widget上"""
        x, y = self.master.winfo_pointerxy()
        widget_x = widget.winfo_rootx()
        widget_y = widget.winfo_rooty()
        widget_width = widget.winfo_width()
        widget_height = widget.winfo_height()
        return widget_x <= x <= widget_x + widget_width and widget_y <= y <= widget_y + widget_height

    def toggle_realtime(self):
        """切换实时识别的启动和停止"""
        if not self.video_running:
            self.start_realtime()
        else:
            self.stop_realtime()

    def start_realtime(self):
        """启动实时识别"""
        self.cap = cv2.VideoCapture(1)  # 打开默认摄像头
        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开摄像头。")
            return

        self.video_running = True
        self.realtime_button.config(text="停止识别", bg="red")
        self.stop_realtime_button.pack(side=tk.LEFT, padx=20)  # 显示停止按钮

        self.video_thread = threading.Thread(target=self.realtime_detection)
        self.video_thread.daemon = True
        self.video_thread.start()
        print("实时识别已启动。")

    def stop_realtime(self):
        """停止实时识别"""
        self.video_running = False
        self.realtime_button.config(text="实时识别", bg="green")
        self.stop_realtime_button.pack_forget()  # 隐藏停止按钮
        if self.cap:
            self.cap.release()
            self.cap = None
        print("实时识别已停止。")

    def realtime_detection(self):
        """实时识别视频流中的对象"""
        while self.video_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取摄像头帧。")
                break

            # 获取当前参数
            dp = self.sliders['dp'].get()
            minDist = self.sliders['minDist'].get()
            param1 = self.sliders['param1'].get()
            param2 = self.sliders['param2'].get()
            minRadius_coin = self.sliders['minRadius_coin'].get()
            maxRadius_coin = self.sliders['maxRadius_coin'].get()
            canny_threshold1 = self.sliders['canny_threshold1'].get()
            canny_threshold2 = self.sliders['canny_threshold2'].get()
            brightness_threshold = self.sliders['brightness_threshold'].get()
            screw_aspect_ratio_min = self.sliders['screw_aspect_ratio_min'].get()
            screw_aspect_ratio_max = self.sliders['screw_aspect_ratio_max'].get()
            nut_min_area = self.sliders['nut_min_area'].get()
            nut_max_area = self.sliders['nut_max_area'].get()
            nut_circularity = self.sliders['nut_circularity'].get()
            screw_min_area = self.sliders['screw_min_area'].get()
            screw_max_area = self.sliders['screw_max_area'].get()
            resize_width = self.sliders['resize_width'].get()
            resize_height = self.sliders['resize_height'].get()

            # 缩放帧
            frame_resized = cv2.resize(frame, (resize_width, resize_height))
            image = frame_resized.copy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 处理光泽反光：检测亮区域并应用模糊
            bright_mask = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)[1]
            bright_mask = cv2.GaussianBlur(bright_mask, (15, 15), 0)
            image_blurred = cv2.GaussianBlur(image, (15, 15), 0)
            image = np.where(bright_mask[:, :, np.newaxis] == 255, image_blurred, image)

            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 应用高斯滤波
            gray_filtered = cv2.GaussianBlur(gray, (9, 9), 2)

            # 融合边缘检测
            edges = self.fused_edge_detection(gray_filtered)

            # 显示边缘检测图
            self.master.after(0, self.display_image, edges, self.edge_canvas, 'edges')

            # 应用形态学闭运算以闭合边缘中的断裂
            kernel = np.ones((3, 3), np.uint8)
            closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

            # 初始化检测计数和检测结果列表
            detections = []
            output = image.copy()

            # 检测硬币
            circles = cv2.HoughCircles(
                closed_edges,
                cv2.HOUGH_GRADIENT,
                dp=dp,
                minDist=minDist,
                param1=param1,
                param2=param2,
                minRadius=minRadius_coin,
                maxRadius=maxRadius_coin
            )

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    area = np.pi * (r ** 2)
                    circularity = 1.0  # 硬币假设为完美圆形
                    if area >= 1000:
                        prob_coin = 1.0
                        detections.append({
                            'type': 'Coin',
                            'bbox': (x - r, y - r, x + r, y + r),
                            'score': prob_coin,
                            'center': (x, y),
                            'radius': r
                        })

            # 检测螺母和螺钉
            contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * (area / (perimeter ** 2))
                x, y, w, h = cv2.boundingRect(cnt)

                # 计算中心
                cx = x + w // 2
                cy = y + h // 2

                # 螺母检测
                if nut_min_area <= area <= nut_max_area and circularity >= nut_circularity:
                    prob_nut = circularity
                    detections.append({
                        'type': 'Nut',
                        'bbox': (x, y, x + w, y + h),
                        'score': prob_nut,
                        'center': (cx, cy),
                        'bbox_size': (w, h)
                    })

                # 螺钉检测
                if screw_min_area <= area <= screw_max_area:
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    width = rect[1][0]
                    height = rect[1][1]
                    if width == 0 or height == 0:
                        continue
                    aspect_ratio = max(width, height) / min(width, height)
                    if screw_aspect_ratio_min <= aspect_ratio <= screw_aspect_ratio_max:
                        prob_screw = 1 / aspect_ratio
                        x_min = min(box[:, 0])
                        y_min = min(box[:, 1])
                        x_max = max(box[:, 0])
                        y_max = max(box[:, 1])
                        detections.append({
                            'type': 'Screw',
                            'bbox': (x_min, y_min, x_max, y_max),
                            'score': prob_screw,
                            'center': (int(rect[0][0]), int(rect[0][1])),
                            'box': box
                        })

            # 处理重叠检测，确保每个位置只有一种类型
            final_detections = self.resolve_overlaps(detections)

            # 重置计数
            coin_count = 0
            nut_count = 0
            screw_count = 0

            # 绘制最终检测结果
            for det in final_detections:
                if det['type'] == 'Coin':
                    x1, y1, x2, y2 = det['bbox']
                    r = det.get('radius', (x2 - x1) // 2)
                    x_center = det['center'][0]
                    y_center = det['center'][1]
                    cv2.circle(output, (x_center, y_center), r, (0, 255, 0), 2)
                    cv2.circle(output, (x_center, y_center), 2, (0, 0, 255), 3)
                    cv2.putText(output, f"Coin {coin_count + 1}", (x_center - r, y_center - r - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    coin_count += 1
                elif det['type'] == 'Nut':
                    x1, y1, x2, y2 = det['bbox']
                    cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(output, f"Nut {nut_count + 1}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    nut_count += 1
                elif det['type'] == 'Screw':
                    box = det['box']
                    cv2.drawContours(output, [box], 0, (0, 0, 255), 2)
                    cv2.putText(output, f"Screw {screw_count + 1}", tuple(box[0]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    screw_count += 1

            # 保存检测到的结果图像以供保存功能使用
            self.detected_image = output.copy()

            # 准备显示结果图像
            # 使用self.master.after确保在主线程中更新GUI
            self.master.after(0, self.update_gui, output, coin_count, nut_count, screw_count)
            print("检测完成，更新GUI。")

if __name__ == "__main__":
    root = tk.Tk()
    app = CoinNutScrewDetectorGUI(root)
    root.mainloop()
