import av
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from sklearn.cluster import AgglomerativeClustering
import subprocess
import logging
import os
import threading
import queue
from tkinter import *
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceSwapApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Target Face Swap")
        self.root.geometry("750x550") # Увеличим окно

        # Глобальный путь больше не нужен для обработки, но оставим для кнопки
        self.initial_target_face_path = "" 
        self.input_video_path = ""
        self.output_dir = os.getcwd()

        self.source_prototypes = []
        self.face_clusters = []
        self.selection_event = threading.Event()

        # --- НОВЫЕ СТРУКТУРЫ ДАННЫХ ---
        # {cluster_idx: 'path/to/target.jpg'}
        self.target_assignments = {} 
        # {face.my_id: cluster_idx}
        self.face_id_to_cluster_index = {}
        # {cluster_idx: insightface_face_object}
        self.processed_target_faces = {}

        self.queue = queue.Queue()
        self.app_analysis = None
        self.swapper = None

        self.create_widgets()
        self.check_queue()

    def create_widgets(self):
        # ... (Код виджетов почти без изменений, только текст) ...
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat")
        style.configure("TLabel", padding=6)
        
        file_frame = ttk.Frame(self.root)
        file_frame.pack(pady=10, fill=X, padx=10)
        
        ttk.Button(file_frame, text="Выбрать видео", command=self.select_video).grid(row=0, column=0, padx=5)
        self.video_label = ttk.Label(file_frame, text="Файл не выбран", width=70)
        self.video_label.grid(row=0, column=1, sticky=W)
        
        # Эта кнопка теперь просто для удобства, можно и не выбирать
        ttk.Button(file_frame, text="Выбрать целевое фото (необязательно)", command=self.select_initial_face).grid(row=1, column=0, padx=5, pady=5)
        self.face_label = ttk.Label(file_frame, text="Назначьте фото в диалоговом окне", width=70)
        self.face_label.grid(row=1, column=1, sticky=W)
        
        self.progress = ttk.Progressbar(self.root, orient=HORIZONTAL, length=550, mode='determinate')
        self.progress.pack(pady=10)
        
        self.status_label = ttk.Label(self.root, text="Готов к работе")
        self.status_label.pack(pady=5)
        
        self.start_btn = ttk.Button(self.root, text="Начать обработку", command=self.start_processing)
        self.start_btn.pack(pady=10)
        
        self.log_text = Text(self.root, height=10, state=DISABLED)
        self.log_text.pack(fill=BOTH, expand=True, padx=10, pady=5)

    def select_initial_face(self):
        # Эта функция теперь просто вспомогательная
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")])
        if file_path:
            self.initial_target_face_path = file_path
            self.face_label.config(text=f"Выбрано по умолчанию: {os.path.basename(file_path)}")

    def reset_ui(self):
        self.progress['value'] = 0
        self.start_btn['state'] = NORMAL
        self.status_label.config(text="Готов к работе")
        self.target_assignments.clear()
        self.face_id_to_cluster_index.clear()
        self.processed_target_faces.clear()

    def start_processing(self):
        if not self.input_video_path:
            messagebox.showerror("Ошибка", "Пожалуйста, выберите видео для обработки!")
            return
        
        self.start_btn['state'] = DISABLED
        self.log_text.delete('1.0', END)
        threading.Thread(target=self.run_full_process, daemon=True).start()

    # --- ЗНАЧИТЕЛЬНО ИЗМЕНЕННАЯ ФУНКЦИЯ ---
    def show_face_selection_dialog(self):
        dialog = Toplevel(self.root)
        dialog.title("Назначьте целевые лица для каждой группы")
        dialog.geometry("800x700")
        dialog.transient(self.root)
        dialog.grab_set()

        # ... (Верхняя часть с ползунком без изменений) ...
        control_frame = ttk.Frame(dialog, padding=10)
        control_frame.pack(fill='x')
        ttk.Label(control_frame, text="Порог группировки:").grid(row=0, column=0, sticky='w')
        threshold_var = DoubleVar(value=20.0)
        threshold_scale = ttk.Scale(control_frame, from_=10.0, to=25.0, orient=HORIZONTAL, variable=threshold_var)
        threshold_scale.grid(row=0, column=1, sticky='ew', padx=5)
        threshold_label = ttk.Label(control_frame, text=f"{threshold_var.get():.2f}", width=5)
        threshold_label.grid(row=0, column=2, sticky='w')
        control_frame.columnconfigure(1, weight=1)
        info_label = ttk.Label(dialog, text="Группировка...", anchor='center')
        info_label.pack(fill='x', padx=10)

        # ... (Часть с canvas без изменений) ...
        outer_canvas_frame = ttk.Frame(dialog)
        outer_canvas_frame.pack(pady=5, padx=10, fill='both', expand=True)
        canvas = Canvas(outer_canvas_frame)
        scrollbar = ttk.Scrollbar(outer_canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        photo_images = []
        checkbox_vars = []
        target_labels = {} # {cluster_idx: label_widget}
        target_paths = {}  # {cluster_idx: 'path/to/img.jpg'}

        def select_target_for_cluster(cluster_idx):
            path = filedialog.askopenfilename(parent=dialog, filetypes=[("Image files", "*.jpg *.jpeg *.png")])
            if path:
                target_paths[cluster_idx] = path
                target_labels[cluster_idx].config(text=os.path.basename(path))

        def update_display(threshold_value):
            for widget in scrollable_frame.winfo_children(): widget.destroy()
            photo_images.clear()
            checkbox_vars.clear()
            target_labels.clear()

            current_clusters = self.cluster_faces(threshold_value)
            info_label.config(text=f"Найдено групп: {len(current_clusters)}")

            NUM_EXAMPLES = 4
            THUMBNAIL_SIZE = (65, 65)

            for i, cluster in enumerate(current_clusters):
                cluster_frame = ttk.Frame(scrollable_frame, borderwidth=2, relief="groove")
                cluster_frame.pack(pady=5, padx=5, fill='x')
                cluster_frame.columnconfigure(1, weight=1)

                # Левая колонка с чекбоксом и картинками
                left_pane = ttk.Frame(cluster_frame)
                left_pane.grid(row=0, column=0, sticky='ns', padx=5, pady=5)
                
                var = BooleanVar()
                checkbox_vars.append(var)
                cb = ttk.Checkbutton(left_pane, text=f"Заменить человека #{i+1}\n({len(cluster)} ракурсов)", variable=var)
                cb.pack(anchor='w')

                image_strip_frame = ttk.Frame(left_pane)
                image_strip_frame.pack(anchor='w', pady=(5,0))
                
                sample_count = min(len(cluster), NUM_EXAMPLES)
                sample_faces = random.sample(cluster, k=sample_count)

                for face_obj in sample_faces:
                    img = Image.fromarray(cv2.cvtColor(face_obj.thumbnail_bgr, cv2.COLOR_BGR2RGB))
                    img.thumbnail(THUMBNAIL_SIZE)
                    photo = ImageTk.PhotoImage(img)
                    photo_images.append(photo)
                    img_label = Label(image_strip_frame, image=photo, borderwidth=1, relief="solid")
                    img_label.pack(side=LEFT, padx=2)

                # Правая колонка для выбора целевого фото
                right_pane = ttk.Frame(cluster_frame)
                right_pane.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)

                # Используем lambda, чтобы передать индекс кластера 'i'
                btn = ttk.Button(right_pane, text="Выбрать фото для замены", command=lambda idx=i: select_target_for_cluster(idx))
                btn.pack(pady=5)
                
                label = ttk.Label(right_pane, text="Фото не выбрано", style="TLabel", anchor="center", justify="center")
                label.pack(pady=5)
                target_labels[i] = label

            dialog.current_clusters = current_clusters
        
        def on_confirm():
            self.face_clusters = getattr(dialog, 'current_clusters', [])
            selected_indices = [i for i, var in enumerate(checkbox_vars) if var.get()]

            if not selected_indices:
                messagebox.showwarning("Нет выбора", "Вы не отметили ни одной группы для замены.", parent=dialog)
                return

            # Проверяем, что для каждой отмеченной группы выбрано целевое фото
            temp_assignments = {}
            for i in selected_indices:
                if i not in target_paths:
                    messagebox.showwarning("Ошибка назначения", f"Для 'Человека #{i+1}' не выбрано фото для замены!", parent=dialog)
                    return
                temp_assignments[i] = target_paths[i]
            
            # Если все проверки пройдены, сохраняем данные в self
            self.target_assignments = temp_assignments
            self.face_id_to_cluster_index.clear()
            for cluster_idx, cluster_data in enumerate(self.face_clusters):
                if cluster_idx in self.target_assignments:
                    for face_obj in cluster_data:
                        self.face_id_to_cluster_index[face_obj.my_id] = cluster_idx
            
            dialog.destroy()
            self.selection_event.set()

        def on_cancel():
            self.target_assignments.clear()
            dialog.destroy()
            self.selection_event.set()

        # ... (Кнопки и биндинги без изменений) ...
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10, fill='x', side=BOTTOM)
        ok_btn = ttk.Button(btn_frame, text="Продолжить", command=on_confirm)
        ok_btn.pack(side=LEFT, expand=True, padx=10)
        cancel_btn = ttk.Button(btn_frame, text="Отмена", command=on_cancel)
        cancel_btn.pack(side=RIGHT, expand=True, padx=10)
        dialog.protocol("WM_DELETE_WINDOW", on_cancel)
        threshold_scale.config(command=lambda v: update_display(threshold_var.get()))
        update_display(threshold_var.get())
        self.root.wait_window(dialog)
        
    # --- ЗНАЧИТЕЛЬНО ИЗМЕНЕННАЯ ФУНКЦИЯ ---
    def run_full_process(self):
        temp_video_path = None
        try:
            self.reset_ui() # Очищаем старые данные
            # ... (Конвертация видео без изменений) ...
            video_for_processing = self.convert_to_mp4(self.input_video_path)
            if video_for_processing != self.input_video_path:
                temp_video_path = video_for_processing

            self.initialize_models()
            self.source_prototypes, frame_to_faces_map = self.analyze_video_and_map_faces(video_for_processing)

            if not self.source_prototypes:
                self.queue.put({'type': 'error', 'message': 'В видео не найдено лиц для анализа.'})
                return

            self.queue.put({'type': 'log', 'message': f'Собрано {len(self.source_prototypes)} образцов лиц для группировки.'})

            # Запускаем диалог выбора
            self.selection_event.clear()
            self.queue.put({'type': 'request_selection'})
            self.selection_event.wait() 

            # Проверяем, были ли сделаны назначения
            if not self.target_assignments:
                self.queue.put({'type': 'log', 'message': 'Обработка отменена: не было сделано ни одного назначения для замены.'})
                return

            # --- НОВЫЙ ШАГ: Предварительная обработка всех целевых лиц ---
            self.queue.put({'type': 'status', 'message': 'Анализ целевых изображений...'})
            self.processed_target_faces.clear()
            for cluster_idx, path in self.target_assignments.items():
                self.queue.put({'type': 'log', 'message': f'Анализ фото: {os.path.basename(path)}'})
                target_image = cv2.imread(path)
                target_faces = self.app_analysis.get(target_image)
                if not target_faces:
                    self.queue.put({'type': 'error', 'message': f'На изображении {os.path.basename(path)} не найдено лиц!'})
                    return
                # Сохраняем обработанный объект лица
                self.processed_target_faces[cluster_idx] = target_faces[0]

            self.queue.put({'type': 'log', 'message': f'Будет заменено {len(self.target_assignments)} групп лиц.'})
            self.queue.put({'type': 'status', 'message': 'Выполняется замена лиц...'})
            self.progress['value'] = 0

            # --- ПРОХОД 2: ЗАМЕНА С НОВОЙ ЛОГИКОЙ ---
            output_path_no_audio = os.path.join(self.output_dir, f'temp_output_no_audio_{random.randint(1000,9999)}.mp4')
            with av.open(video_for_processing) as input_container, av.open(output_path_no_audio, mode='w') as output_container:
                # ... (Настройка потоков без изменений) ...
                input_stream = input_container.streams.video[0]
                output_stream = output_container.add_stream('libx264', rate=input_stream.average_rate)
                output_stream.width = input_stream.width
                output_stream.height = input_stream.height
                output_stream.pix_fmt = 'yuv420p'
                output_stream.options = {'crf': '18', 'preset': 'fast'}
                total_frames = input_stream.frames if input_stream.frames > 0 else 1

                for i, frame in enumerate(input_container.decode(input_stream)):
                    img = frame.to_ndarray(format='bgr24')
                    faces_in_frame = frame_to_faces_map.get(i, [])

                    for source_face in faces_in_frame:
                        # Проверяем, нужно ли заменять это лицо
                        if source_face.my_id in self.face_id_to_cluster_index:
                            # Находим, к какому кластеру оно относится
                            cluster_idx = self.face_id_to_cluster_index[source_face.my_id]
                            # Берем соответствующее ему целевое лицо
                            target_face_for_swap = self.processed_target_faces[cluster_idx]
                            # Выполняем замену
                            img = self.swapper.get(img, source_face, target_face_for_swap)
                    
                    # ... (Кодирование кадра и прогресс без изменений) ...
                    output_frame = av.VideoFrame.from_ndarray(img, format='bgr24')
                    for packet in output_stream.encode(output_frame):
                        output_container.mux(packet)
                    progress = (i / total_frames) * 100
                    self.queue.put({'type': 'progress', 'value': progress})
                
                for packet in output_stream.encode():
                    output_container.mux(packet)

            # ... (Слияние с аудио и очистка без изменений) ...
            self.queue.put({'type': 'status', 'message': 'Добавление аудиодорожки...'})
            base_name = os.path.splitext(os.path.basename(self.input_video_path))[0]
            final_output = os.path.join(self.output_dir, f'final_output_{base_name}.mp4')
            command = ['ffmpeg', '-y', '-i', output_path_no_audio, '-i', video_for_processing, '-c:v', 'copy', '-c:a', 'copy', '-map', '0:v:0', '-map', '1:a:0?', '-shortest', final_output]
            subprocess.run(command, check=True, capture_output=True, text=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))

            self.queue.put({'type': 'done'})
            self.queue.put({'type': 'log', 'message': f'Видео успешно сохранено в: {final_output}'})
            os.remove(output_path_no_audio)

        except Exception as e:
            error_details = str(e)
            if isinstance(e, subprocess.CalledProcessError):
                error_details += f"\nFFMPEG stderr: {e.stderr}"
            self.queue.put({'type': 'error', 'message': f'Ошибка обработки: {error_details}'})
            logger.error(f"Полная ошибка: {e}", exc_info=True)
        finally:
            self.reset_ui() # Сброс UI и очистка данных
            if not self.selection_event.is_set():
                self.selection_event.set()
            if temp_video_path and os.path.exists(temp_video_path):
                try: os.remove(temp_video_path)
                except OSError as e_remove: self.queue.put({'type': 'log', 'message': f'Не удалось удалить временный файл {temp_video_path}: {e_remove}'})

    # Оставим остальные функции без изменений
    def log_message(self, message):
        self.log_text.configure(state=NORMAL)
        self.log_text.insert(END, message + "\n")
        self.log_text.see(END)
        self.log_text.configure(state=DISABLED)

    def check_queue(self):
        while not self.queue.empty():
            try:
                msg = self.queue.get_nowait()
                if msg['type'] == 'progress': self.progress['value'] = msg['value']
                elif msg['type'] == 'log': self.log_message(msg['message'])
                elif msg['type'] == 'status': self.status_label.config(text=msg['message'])
                elif msg['type'] == 'error': messagebox.showerror("Ошибка", msg['message']); self.reset_ui()
                elif msg['type'] == 'done': messagebox.showinfo("Успех", "Обработка завершена!"); self.reset_ui()
                elif msg['type'] == 'request_selection': self.show_face_selection_dialog()
            except queue.Empty: pass
        self.root.after(100, self.check_queue)

    def select_video(self, *args):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.webm"), ("All files", "*.*")])
        if file_path: self.input_video_path = file_path; self.video_label.config(text=os.path.basename(file_path))

    def initialize_models(self):
        if self.app_analysis is None:
            self.queue.put({'type': 'status', 'message': 'Загрузка модели анализа лиц...'}); self.app_analysis = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider']); self.app_analysis.prepare(ctx_id=0, det_size=(640, 640))
        if self.swapper is None:
            self.queue.put({'type': 'status', 'message': 'Загрузка модели замены лиц...'}); self.swapper = insightface.model_zoo.get_model('inswapper_128.onnx', providers=['CUDAExecutionProvider'])

    def convert_to_mp4(self, video_path):
        file_name, file_extension = os.path.splitext(video_path)
        if file_extension.lower() == '.mp4': self.queue.put({'type': 'log', 'message': 'Видео уже в формате MP4. Конвертация не требуется.'}); return video_path
        self.queue.put({'type': 'status', 'message': f'Конвертация {file_extension} в MP4...'}); self.queue.put({'type': 'log', 'message': 'Начало конвертации видео в стандартный формат MP4.'})
        temp_output_path = os.path.join(self.output_dir, f"temp_converted_{random.randint(1000, 9999)}.mp4")
        try:
            command = ['ffmpeg', '-y', '-i', video_path, '-c:v', 'libx264', '-preset', 'fast', '-crf', '22', '-c:a', 'aac', '-b:a', '192k', '-pix_fmt', 'yuv420p', temp_output_path]
            subprocess.run(command, check=True, capture_output=True, text=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            self.queue.put({'type': 'log', 'message': 'Конвертация в MP4 успешно завершена.'}); return temp_output_path
        except subprocess.CalledProcessError as e: raise RuntimeError(f"Ошибка конвертации видео: {e.stderr}")

    def analyze_video_and_map_faces(self, video_path):
        self.queue.put({'type': 'status', 'message': 'Полный анализ видео (может быть долго)...'}); all_prototypes, frame_to_faces_map, face_counter = [], {}, 0
        with av.open(video_path) as container:
            stream = container.streams.video[0]; stream.thread_type = "AUTO"; total_frames = stream.frames if stream.frames > 0 else 1
            for i, frame in enumerate(container.decode(stream)):
                frame_to_faces_map[i] = []; img = frame.to_ndarray(format='bgr24'); detected_faces = self.app_analysis.get(img)
                for face in detected_faces:
                    bbox = face.bbox.astype(int)
                    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]: continue
                    padding = 20; face_crop = img[max(0, bbox[1] - padding):min(img.shape[0], bbox[3] + padding), max(0, bbox[0] - padding):min(img.shape[1], bbox[2] + padding)]
                    if face_crop.size == 0: continue
                    face.thumbnail_bgr = face_crop; face.my_id = face_counter; all_prototypes.append(face); frame_to_faces_map[i].append(face); face_counter += 1
                self.queue.put({'type': 'progress', 'value': (i / total_frames) * 100})
        self.queue.put({'type': 'progress', 'value': 100}); return all_prototypes, frame_to_faces_map

    def cluster_faces(self, threshold):
        if len(self.source_prototypes) < 2: return [[p] for p in self.source_prototypes]
        embeddings = np.array([face.embedding for face in self.source_prototypes])
        clustering = AgglomerativeClustering(n_clusters=None, metric='euclidean', linkage='average', distance_threshold=threshold); clustering.fit(embeddings); labels = clustering.labels_
        num_clusters = len(set(labels) - {-1}); clusters = [[] for _ in range(num_clusters)] 
        for i, face_obj in enumerate(self.source_prototypes):
            if labels[i] != -1: clusters[labels[i]].append(face_obj)
        clusters = [c for c in clusters if c]; clusters.sort(key=len, reverse=True); return clusters

if __name__ == "__main__":
    root = Tk()
    app = FaceSwapApp(root)
    root.mainloop()