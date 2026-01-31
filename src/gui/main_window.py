"""
Content-Based Image Retrieval System - GUI
Based on Shahid Beheshti University DS&A Project
WITH BONUS FEATURES: HNSW + Complete Comparison + Dynamic Method Selection
"""

import sys
import os
import numpy as np
from pathlib import Path
import time

# اضافه کردن path پروژه
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# ابتدا ماژول‌های torch را import کنید (قبل از PyQt)
from utils.image_processor import ImageProcessor
from core.vector_db import VectorDatabase

# حالا PyQt را import کنید
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QScrollArea, QGridLayout,
    QComboBox, QSpinBox, QProgressBar, QTabWidget, QTextEdit,
    QGroupBox, QMessageBox, QLineEdit, QTableWidget, QTableWidgetItem
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap

# بقیه import ها
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import logging


class SearchThread(QThread):
    """✅ Thread برای جستجوی غیرهمزمان (با پشتیبانی HNSW)"""
    finished = pyqtSignal(list, float, str)  # (results, time_taken, method)
    error = pyqtSignal(str)

    def __init__(self, vector_db, query_vector, top_k, search_method):
        """
        Args:
            search_method: 'lsh', 'brute-force', یا 'hnsw'
        """
        super().__init__()
        self.vector_db = vector_db
        self.query_vector = query_vector
        self.top_k = top_k
        self.search_method = search_method

    def run(self):
        try:
            start_time = time.time()

            if self.search_method == 'hnsw':
                results = self.vector_db.find_similar(
                    self.query_vector,
                    top_k=self.top_k,
                    use_lsh=False,
                    use_hnsw=True
                )
                method = "HNSW (BONUS)"
            elif self.search_method == 'lsh':
                results = self.vector_db.find_similar(
                    self.query_vector,
                    top_k=self.top_k,
                    use_lsh=True,
                    use_hnsw=False
                )
                method = "LSH"
            else:  # brute-force
                results = self.vector_db.find_similar(
                    self.query_vector,
                    top_k=self.top_k,
                    use_lsh=False,
                    use_hnsw=False
                )
                method = "Brute-force"

            elapsed = time.time() - start_time
            self.finished.emit(results, elapsed, method)

        except Exception as e:
            self.error.emit(str(e))


class EmbeddingCanvas(FigureCanvas):
    """Canvas برای نمایش scatter plot embeddings"""

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

    def plot_embeddings(self, vector_db, query_id=None):
        """نمایش 2D projection از embeddings با PCA"""
        self.axes.clear()

        if len(vector_db.vectors) == 0:
            self.axes.text(0.5, 0.5, 'No data in database',
                           ha='center', va='center', fontsize=14)
            self.draw()
            return

        vectors = []
        categories = []
        ids = []

        for vec_id, vector in vector_db.vectors.items():
            vectors.append(vector)
            metadata = vector_db.metadata.get(vec_id, {})
            category = metadata.get('category', 'unknown')
            categories.append(category)
            ids.append(vec_id)

        vectors = np.array(vectors)

        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(vectors)

        unique_categories = list(set(categories))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_categories)))
        category_colors = {cat: colors[i] for i, cat in enumerate(unique_categories)}

        for i, category in enumerate(unique_categories):
            mask = np.array(categories) == category
            self.axes.scatter(
                vectors_2d[mask, 0],
                vectors_2d[mask, 1],
                c=[category_colors[category]],
                label=category,
                alpha=0.6,
                s=50
            )

        if query_id and query_id in ids:
            query_idx = ids.index(query_id)
            self.axes.scatter(
                vectors_2d[query_idx, 0],
                vectors_2d[query_idx, 1],
                c='red',
                marker='*',
                s=500,
                edgecolors='black',
                linewidths=2,
                label='Query',
                zorder=5
            )

        self.axes.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=10)
        self.axes.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=10)
        self.axes.set_title('Embedding Space Visualization (PCA)', fontsize=12, fontweight='bold')
        self.axes.legend(loc='best', fontsize=8)
        self.axes.grid(True, alpha=0.3)
        self.draw()


class ImageLabel(QLabel):
    """Label برای نمایش تصویر با اطلاعات"""

    def __init__(self, image_path, similarity=None, vec_id=None):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 2px solid #ccc; border-radius: 5px; padding: 5px;")
        self.setFixedSize(200, 250)

        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(180, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(pixmap)

        if similarity is not None:
            text = f"{vec_id}\nSimilarity: {similarity:.4f}"
            self.setText(text)
            self.setWordWrap(True)


class CBIRMainWindow(QMainWindow):
    """پنجره اصلی سیستم جستجوی تصویر (با قابلیت‌های بونوس)"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CBIR using LSH + HNSW (BONUS) - SBU DS&A Project")
        self.setGeometry(100, 100, 1400, 900)

        self.vector_db = None
        self.image_processor = None
        self.query_image_path = None
        self.query_embedding = None

        # مسیر دیتابیس
        self.db_path = str(PROJECT_ROOT / "data" / "embeddings" / "caltech101_db.pkl")

        self.init_ui()
        self.initialize_database()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        search_tab = QWidget()
        tabs.addTab(search_tab, "🔍 Image Search")
        self.setup_search_tab(search_tab)

        perf_tab = QWidget()
        tabs.addTab(perf_tab, "⚡ Performance Analysis")
        self.setup_performance_tab(perf_tab)

        # ✅ BONUS: تب مقایسه کامل
        bonus_tab = QWidget()
        tabs.addTab(bonus_tab, "🎁 BONUS: Complete Comparison")
        self.setup_bonus_tab(bonus_tab)

        viz_tab = QWidget()
        tabs.addTab(viz_tab, "📊 Embedding Visualization")
        self.setup_visualization_tab(viz_tab)

        stats_tab = QWidget()
        tabs.addTab(stats_tab, "📈 Database Statistics")
        self.setup_stats_tab(stats_tab)

        self.statusBar().showMessage("Ready")

    def setup_search_tab(self, parent):
        layout = QVBoxLayout(parent)

        control_layout = QHBoxLayout()

        self.select_btn = QPushButton("📁 Select Query Image")
        self.select_btn.clicked.connect(self.select_query_image)
        self.select_btn.setStyleSheet("font-size: 14px; padding: 10px;")
        control_layout.addWidget(self.select_btn)

        self.top_k_label = QLabel("Top K:")
        control_layout.addWidget(self.top_k_label)

        self.top_k_spinbox = QSpinBox()
        self.top_k_spinbox.setRange(1, 50)
        self.top_k_spinbox.setValue(10)
        control_layout.addWidget(self.top_k_spinbox)

        # ✅ فقط پیش‌فرض‌ها (HNSW بعداً اضافه میشه)
        self.search_method_combo = QComboBox()
        self.search_method_combo.addItems(["LSH (Fast)", "Brute-force (Exact)"])
        control_layout.addWidget(self.search_method_combo)

        self.search_btn = QPushButton("🔍 Search Similar Images")
        self.search_btn.clicked.connect(self.search_similar_images)
        self.search_btn.setEnabled(False)
        self.search_btn.setStyleSheet(
            "font-size: 14px; padding: 10px; background-color: #4CAF50; color: white;"
        )
        control_layout.addWidget(self.search_btn)

        control_layout.addStretch()
        layout.addLayout(control_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.search_time_label = QLabel("")
        self.search_time_label.setStyleSheet("font-size: 12px; color: #0066cc; padding: 5px;")
        layout.addWidget(self.search_time_label)

        query_group_layout = QHBoxLayout()
        self.query_image_label = QLabel()
        self.query_image_label.setFixedSize(300, 300)
        self.query_image_label.setAlignment(Qt.AlignCenter)
        self.query_image_label.setStyleSheet(
            "border: 3px solid #2196F3; border-radius: 10px; background-color: #f0f0f0;"
        )
        self.query_image_label.setText("No query image selected")
        query_group_layout.addWidget(self.query_image_label)
        query_group_layout.addStretch()
        layout.addLayout(query_group_layout)

        results_label = QLabel("Search Results:")
        results_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 10px;")
        layout.addWidget(results_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: white;")
        self.results_widget = QWidget()
        self.results_layout = QGridLayout(self.results_widget)
        self.results_layout.setSpacing(15)
        scroll.setWidget(self.results_widget)
        layout.addWidget(scroll)

    def setup_performance_tab(self, parent):
        layout = QVBoxLayout(parent)

        compare_btn = QPushButton("⚡ Run Performance Comparison (LSH vs Brute-force)")
        compare_btn.setStyleSheet("font-size: 14px; padding: 10px; background-color: #FF9800; color: white;")
        compare_btn.clicked.connect(self.run_performance_comparison)
        layout.addWidget(compare_btn)

        self.perf_table = QTableWidget()
        self.perf_table.setColumnCount(4)
        self.perf_table.setHorizontalHeaderLabels(["Method", "Time (ms)", "Results Found", "Speed Ratio"])
        layout.addWidget(self.perf_table)

        self.perf_canvas = FigureCanvas(Figure(figsize=(10, 6)))
        layout.addWidget(self.perf_canvas)

    def setup_bonus_tab(self, parent):
        """✅ BONUS: تب مقایسه کامل همه الگوریتم‌ها"""
        layout = QVBoxLayout(parent)

        # دکمه مقایسه
        compare_btn = QPushButton("🎁 Compare ALL Algorithms (Brute-force + LSH + HNSW)")
        compare_btn.setStyleSheet(
            "font-size: 14px; padding: 12px; background-color: #9C27B0; color: white; font-weight: bold;"
        )
        compare_btn.clicked.connect(self.run_complete_comparison)
        layout.addWidget(compare_btn)

        # اطلاعات
        info_label = QLabel(
            "🎯 This feature compares all available search methods:\n"
            "• Brute-force (Ground Truth)\n"
            "• LSH (Required)\n"
            "• HNSW (BONUS) - if available\n\n"
            "Select a query image first!"
        )
        info_label.setStyleSheet("font-size: 11px; padding: 10px; background-color: #FFF3E0; border-radius: 5px;")
        layout.addWidget(info_label)

        # جدول نتایج
        self.bonus_table = QTableWidget()
        self.bonus_table.setColumnCount(5)
        self.bonus_table.setHorizontalHeaderLabels([
            "Algorithm", "Time (ms)", "Speedup", "Precision@10", "Status"
        ])
        self.bonus_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.bonus_table)

        # نمودار مقایسه
        self.bonus_canvas = FigureCanvas(Figure(figsize=(12, 6)))
        layout.addWidget(self.bonus_canvas)

    def setup_visualization_tab(self, parent):
        layout = QVBoxLayout(parent)

        control_layout = QHBoxLayout()
        refresh_btn = QPushButton("🔄 Refresh Plot")
        refresh_btn.clicked.connect(self.refresh_embedding_plot)
        refresh_btn.setStyleSheet("padding: 8px; font-size: 13px;")
        control_layout.addWidget(refresh_btn)
        control_layout.addStretch()
        layout.addLayout(control_layout)

        self.embedding_canvas = EmbeddingCanvas(self, width=10, height=7)
        layout.addWidget(self.embedding_canvas)

    def setup_stats_tab(self, parent):
        layout = QVBoxLayout(parent)

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setStyleSheet("font-family: 'Courier New'; font-size: 12px;")
        layout.addWidget(self.stats_text)

        refresh_stats_btn = QPushButton("🔄 Refresh Statistics")
        refresh_stats_btn.clicked.connect(self.update_stats)
        refresh_stats_btn.setStyleSheet("padding: 10px; font-size: 13px;")
        layout.addWidget(refresh_stats_btn)

    def _update_search_methods_combo(self):
        """
        ✅ به‌روزرسانی گزینه‌های جستجو بر اساس قابلیت‌های database
        """
        # ذخیره انتخاب فعلی
        current_selection = self.search_method_combo.currentIndex()

        # پاک کردن و اضافه مجدد
        self.search_method_combo.clear()

        methods = ["LSH (Fast)", "Brute-force (Exact)"]

        # ✅ BONUS: اگر HNSW فعال باشه
        if self.vector_db and self.vector_db.use_hnsw:
            methods.append("🎁 HNSW (BONUS)")
            logging.info("✅ HNSW search method added to UI")

        self.search_method_combo.addItems(methods)

        # بازگرداندن انتخاب قبلی (یا LSH به عنوان default)
        if current_selection < len(methods):
            self.search_method_combo.setCurrentIndex(current_selection)
        else:
            self.search_method_combo.setCurrentIndex(0)  # LSH

    def initialize_database(self):
        """بارگذاری دیتابیس (با پشتیبانی HNSW)"""
        try:
            self.statusBar().showMessage(f"Initializing database from {self.db_path}...")

            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

            # ✅ فعال‌سازی HNSW (BONUS)
            self.vector_db = VectorDatabase(
                dim=512,
                persist_path=self.db_path,
                use_lsh=True,
                use_hnsw=True,  # ✅ تغییر به True برای فعال‌سازی HNSW
                lsh_params={'num_tables': 8, 'hash_size': 10, 'seed': 42},
                hnsw_params={'M': 16, 'ef_construction': 200}
            )

            self.image_processor = ImageProcessor(device='auto')

            # ✅ به‌روزرسانی ComboBox بعد از ساخت database
            self._update_search_methods_combo()

            num_images = len(self.vector_db.vectors)
            hnsw_status = "✅ HNSW Enabled" if self.vector_db.use_hnsw else "⚠️ HNSW Disabled"

            self.statusBar().showMessage(
                f"Database loaded: {num_images} images | {hnsw_status}"
            )

            if num_images > 0:
                self.update_stats()
                self.refresh_embedding_plot()

        except Exception as e:
            self.statusBar().showMessage(f"Error initializing database: {str(e)}")
            logging.error(f"Database initialization error: {e}")

    def select_query_image(self):
        """انتخاب تصویر query"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Query Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            self.query_image_path = file_path

            pixmap = QPixmap(file_path)
            pixmap = pixmap.scaled(280, 280, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.query_image_label.setPixmap(pixmap)

            try:
                self.query_embedding = self.image_processor.process_image(file_path)
                self.search_btn.setEnabled(True)
                self.statusBar().showMessage(f"Query image loaded: {Path(file_path).name}")
            except Exception as e:
                self.statusBar().showMessage(f"Error processing image: {str(e)}")
                self.search_btn.setEnabled(False)

    def search_similar_images(self):
        """جستجوی تصاویر مشابه (با پشتیبانی HNSW)"""
        if self.query_embedding is None:
            return

        if len(self.vector_db.vectors) == 0:
            QMessageBox.warning(self, "Warning", "Database is empty!")
            return

        # پاک کردن نتایج قبلی
        for i in reversed(range(self.results_layout.count())):
            self.results_layout.itemAt(i).widget().setParent(None)

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.search_btn.setEnabled(False)

        top_k = self.top_k_spinbox.value()

        # تعیین روش جستجو
        method_text = self.search_method_combo.currentText()

        if "LSH" in method_text:
            search_method = 'lsh'
        elif "Brute-force" in method_text:
            search_method = 'brute-force'
        else:  # HNSW
            search_method = 'hnsw'

        self.search_thread = SearchThread(
            self.vector_db,
            self.query_embedding,
            top_k,
            search_method
        )
        self.search_thread.finished.connect(self.display_results)
        self.search_thread.error.connect(self.search_error)
        self.search_thread.start()

    def display_results(self, results, elapsed, method):
        """نمایش نتایج جستجو"""
        self.progress_bar.setVisible(False)
        self.search_btn.setEnabled(True)

        self.search_time_label.setText(
            f"🕐 Search completed using {method} in {elapsed * 1000:.2f} ms | Found {len(results)} images"
        )

        if not results:
            no_result_label = QLabel("No results found")
            no_result_label.setStyleSheet("font-size: 16px; color: red;")
            self.results_layout.addWidget(no_result_label, 0, 0)
            return

        cols = 5
        for idx, (vec_id, similarity) in enumerate(results):
            row = idx // cols
            col = idx % cols

            metadata = self.vector_db.get_metadata(vec_id)
            rel_path = metadata.get('image_path', '')

            rel_path_norm = rel_path.replace("\\", "/")
            parts = rel_path_norm.split("/")
            if parts[0].lower().startswith("caltech101"):
                rel_rel = "/".join(parts[1:])
            else:
                rel_rel = rel_path_norm

            base_dir = PROJECT_ROOT / "data" / "caltech101" / "images" / "101_ObjectCategories"
            full_path = (base_dir / rel_rel).resolve()

            if full_path.exists():
                img_label = ImageLabel(str(full_path), similarity, vec_id)
                self.results_layout.addWidget(img_label, row, col)

        self.statusBar().showMessage(f"Found {len(results)} similar images using {method}")

    def search_error(self, error_msg):
        """مدیریت خطا در جستجو"""
        self.progress_bar.setVisible(False)
        self.search_btn.setEnabled(True)
        self.statusBar().showMessage(f"Search error: {error_msg}")
        QMessageBox.critical(self, "Search Error", f"An error occurred during search:\n{error_msg}")

    def run_performance_comparison(self):
        """مقایسه عملکرد LSH و Brute-force (تب عملکرد)"""
        if len(self.vector_db.vectors) == 0:
            QMessageBox.warning(self, "Warning", "Database is empty!")
            return

        if self.query_embedding is None:
            QMessageBox.warning(self, "Warning", "Please select a query image first!")
            return

        try:
            self.statusBar().showMessage("Running performance comparison...")

            start = time.time()
            bf_results = self.vector_db.find_similar(self.query_embedding, top_k=10, use_lsh=False)
            bf_time = (time.time() - start) * 1000

            start = time.time()
            lsh_results = self.vector_db.find_similar(self.query_embedding, top_k=10, use_lsh=True)
            lsh_time = (time.time() - start) * 1000

            speed_ratio = bf_time / lsh_time if lsh_time > 0 else 0

            self.perf_table.setRowCount(2)

            self.perf_table.setItem(0, 0, QTableWidgetItem("Brute-force (Exact)"))
            self.perf_table.setItem(0, 1, QTableWidgetItem(f"{bf_time:.2f}"))
            self.perf_table.setItem(0, 2, QTableWidgetItem(str(len(bf_results))))
            self.perf_table.setItem(0, 3, QTableWidgetItem("1.0x (baseline)"))

            self.perf_table.setItem(1, 0, QTableWidgetItem("LSH (Approximate)"))
            self.perf_table.setItem(1, 1, QTableWidgetItem(f"{lsh_time:.2f}"))
            self.perf_table.setItem(1, 2, QTableWidgetItem(str(len(lsh_results))))
            self.perf_table.setItem(1, 3, QTableWidgetItem(f"{speed_ratio:.2f}x faster"))

            ax = self.perf_canvas.figure.clear()
            ax = self.perf_canvas.figure.add_subplot(111)

            methods = ['Brute-force', 'LSH']
            times = [bf_time, lsh_time]
            colors = ['#FF6B6B', '#4ECDC4']

            bars = ax.bar(methods, times, color=colors, alpha=0.7)
            ax.set_ylabel('Time (ms)', fontsize=12)
            ax.set_title('Performance Comparison: Brute-force vs LSH', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

            for bar, time_val in zip(bars, times):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{time_val:.2f} ms',
                        ha='center', va='bottom', fontsize=10)

            self.perf_canvas.draw()

            self.statusBar().showMessage(
                f"Comparison complete: LSH is {speed_ratio:.2f}x faster than Brute-force"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Performance comparison failed:\n{str(e)}")

    def run_complete_comparison(self):
        """✅ BONUS: مقایسه کامل همه الگوریتم‌ها"""
        if self.query_embedding is None:
            QMessageBox.warning(self, "Warning", "Please select a query image first!")
            return

        if len(self.vector_db.vectors) == 0:
            QMessageBox.warning(self, "Warning", "Database is empty!")
            return

        try:
            self.statusBar().showMessage("Running complete comparison (including BONUS)...")

            results_data = []

            # 1. Brute-force (Ground Truth)
            start = time.time()
            bf_results = self.vector_db.find_similar(
                self.query_embedding, top_k=10, use_lsh=False, use_hnsw=False
            )
            bf_time = (time.time() - start) * 1000
            bf_ids = set(vid for vid, _ in bf_results)

            results_data.append({
                'name': 'Brute-force',
                'time': bf_time,
                'results': bf_results,
                'color': '#FF6B6B'
            })

            # 2. LSH
            start = time.time()
            lsh_results = self.vector_db.find_similar(
                self.query_embedding, top_k=10, use_lsh=True, use_hnsw=False
            )
            lsh_time = (time.time() - start) * 1000
            lsh_ids = set(vid for vid, _ in lsh_results)
            lsh_precision = len(lsh_ids & bf_ids) / 10 * 100 if lsh_results else 0

            results_data.append({
                'name': 'LSH',
                'time': lsh_time,
                'results': lsh_results,
                'precision': lsh_precision,
                'speedup': bf_time / lsh_time if lsh_time > 0 else 0,
                'color': '#4ECDC4'
            })

            # 3. HNSW (اگر فعال باشد)
            if self.vector_db.use_hnsw and self.vector_db.hnsw_index:
                start = time.time()
                hnsw_results = self.vector_db.find_similar(
                    self.query_embedding, top_k=10, use_hnsw=True
                )
                hnsw_time = (time.time() - start) * 1000
                hnsw_ids = set(vid for vid, _ in hnsw_results)
                hnsw_precision = len(hnsw_ids & bf_ids) / 10 * 100 if hnsw_results else 0

                results_data.append({
                    'name': 'HNSW (BONUS)',
                    'time': hnsw_time,
                    'results': hnsw_results,
                    'precision': hnsw_precision,
                    'speedup': bf_time / hnsw_time if hnsw_time > 0 else 0,
                    'color': '#9C27B0'
                })

            # پر کردن جدول
            self.bonus_table.setRowCount(len(results_data))

            for i, data in enumerate(results_data):
                self.bonus_table.setItem(i, 0, QTableWidgetItem(data['name']))
                self.bonus_table.setItem(i, 1, QTableWidgetItem(f"{data['time']:.2f}"))

                speedup = data.get('speedup', 1.0)
                self.bonus_table.setItem(i, 2, QTableWidgetItem(f"{speedup:.2f}x"))

                precision = data.get('precision', 100.0)
                self.bonus_table.setItem(i, 3, QTableWidgetItem(f"{precision:.1f}%"))

                status = "✅ Ground Truth" if i == 0 else f"✅ {precision:.0f}% Accurate"
                self.bonus_table.setItem(i, 4, QTableWidgetItem(status))

            # رسم نمودار
            self._plot_complete_comparison(results_data)

            self.statusBar().showMessage("✅ Complete comparison finished!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Complete comparison failed:\n{str(e)}")
            logging.error(f"Comparison error: {e}")

    def _plot_complete_comparison(self, results_data):
        """رسم نمودار مقایسه کامل"""
        fig = self.bonus_canvas.figure
        fig.clear()

        # دو subplot: زمان و دقت
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        names = [d['name'] for d in results_data]
        times = [d['time'] for d in results_data]
        colors = [d['color'] for d in results_data]

        # نمودار زمان
        bars1 = ax1.bar(names, times, color=colors, alpha=0.7)
        ax1.set_ylabel('Time (ms)', fontsize=11)
        ax1.set_title('Search Time Comparison', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        for bar, time_val in zip(bars1, times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{time_val:.1f}ms',
                     ha='center', va='bottom', fontsize=9)

        # نمودار دقت
        precisions = [d.get('precision', 100.0) for d in results_data]
        bars2 = ax2.bar(names, precisions, color=colors, alpha=0.7)
        ax2.set_ylabel('Precision@10 (%)', fontsize=11)
        ax2.set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 105)
        ax2.grid(axis='y', alpha=0.3)

        for bar, prec in zip(bars2, precisions):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{prec:.1f}%',
                     ha='center', va='bottom', fontsize=9)

        fig.tight_layout()
        self.bonus_canvas.draw()

    def refresh_embedding_plot(self):
        """بهروزرسانی نمودار embeddings"""
        if self.vector_db and len(self.vector_db.vectors) > 0:
            try:
                self.embedding_canvas.plot_embeddings(self.vector_db)
                self.statusBar().showMessage("Embedding plot updated")
            except Exception as e:
                self.statusBar().showMessage(f"Error plotting embeddings: {str(e)}")

    def update_stats(self):
        """بهروزرسانی آمار دیتابیس (با HNSW)"""
        if not self.vector_db:
            return

        stats = self.vector_db.get_database_stats()

        stats_text = "=" * 60 + "\n"
        stats_text += "DATABASE STATISTICS\n"
        stats_text += "=" * 60 + "\n\n"
        stats_text += f"Total Vectors: {stats['total_vectors']}\n"
        stats_text += f"Dimension: {stats['dimension']}\n"
        stats_text += f"LSH Enabled: {stats['use_lsh']}\n"
        stats_text += f"HNSW Enabled: {stats.get('use_hnsw', False)} {'🎁 (BONUS)' if stats.get('use_hnsw') else ''}\n"
        stats_text += f"Thread-Safe: {stats.get('thread_safe', False)} {'🔒 (BONUS)' if stats.get('thread_safe') else ''}\n"
        stats_text += f"Storage Path: {stats['persist_path']}\n\n"

        # LSH Stats
        if 'lsh_stats' in stats:
            stats_text += "-" * 60 + "\n"
            stats_text += "LSH INDEX STATISTICS\n"
            stats_text += "-" * 60 + "\n"
            lsh_stats = stats['lsh_stats']
            for key, value in lsh_stats.items():
                stats_text += f"  {key}: {value}\n"
            stats_text += "\n"

        # ✅ BONUS: HNSW Stats
        if 'hnsw_stats' in stats:
            stats_text += "-" * 60 + "\n"
            stats_text += "🎁 HNSW INDEX STATISTICS (BONUS)\n"
            stats_text += "-" * 60 + "\n"
            hnsw_stats = stats['hnsw_stats']
            for key, value in hnsw_stats.items():
                stats_text += f"  {key}: {value}\n"
            stats_text += "\n"

        # Category Distribution
        categories = {}
        for vec_id in self.vector_db.get_all_ids():
            metadata = self.vector_db.get_metadata(vec_id)
            cat = metadata.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1

        if categories:
            stats_text += "-" * 60 + "\n"
            stats_text += "CATEGORY DISTRIBUTION\n"
            stats_text += "-" * 60 + "\n"
            for cat, count in sorted(categories.items()):
                stats_text += f"  {cat}: {count} images\n"

        self.stats_text.setText(stats_text)


def main():
    """تابع اصلی برای اجرای GUI"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = CBIRMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
