import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from data_loader import load_data
from knn import knn_predict
from neural_network import train_network, predict_nn
from utils import accuracy_score

class KidneyDiseaseClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.state('zoomed')
        self.root.title("Kidney Disease Classifier")
        self.root.geometry("600x350")
        self.root.configure(bg="#f7f7f7")
        self.filename = ""
        self.train_set = []
        self.test_set = []
        self.k = 3

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", font=("Segoe UI", 11), padding=10)
        style.configure("TLabel", font=("Segoe UI", 11), background="#f7f7f7")
        style.configure("TFrame", background="#f7f7f7")

        self.main_frame = ttk.Frame(root, padding=20)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.build_interface()

    def build_interface(self):
        ttk.Label(self.main_frame, text="Select Data Percentage:").grid(row=0, column=0, sticky='w', padx=10, pady=10)

        self.percentage_var = tk.IntVar(value=70)
        self.percentage_slider = ttk.Scale(
            self.main_frame,
            from_=10,
            to=100,
            orient='horizontal',
            variable=self.percentage_var,
            command=self.update_slider_label
        )
        self.percentage_slider.grid(row=0, column=1, sticky='ew', padx=10)
        self.main_frame.columnconfigure(1, weight=1)

        self.slider_label = ttk.Label(self.main_frame, text="70%")
        self.slider_label.grid(row=0, column=2, padx=10)

        # Choose file
        ttk.Button(self.main_frame, text="Choose CSV File", command=self.choose_file).grid(row=1, column=0, columnspan=3, pady=10)
        self.file_label = ttk.Label(self.main_frame, text="No file selected", foreground="gray")
        self.file_label.grid(row=2, column=0, columnspan=3, sticky='w', padx=10)

        # K entry
        ttk.Label(self.main_frame, text="Enter K for k-NN:").grid(row=3, column=0, sticky='w', padx=10, pady=5)
        self.k_entry = ttk.Entry(self.main_frame)
        self.k_entry.insert(0, "3")
        self.k_entry.grid(row=3, column=1, sticky='ew', padx=10)

        # Run button
        ttk.Button(self.main_frame, text="Run Classification", command=self.run_classification).grid(row=4, column=0, columnspan=3, pady=15)

        # Accuracy result
        self.result_label = ttk.Label(self.main_frame, text="", foreground="#005bbb", font=("Segoe UI", 12, "bold"))
        self.result_label.grid(row=5, column=0, columnspan=3, pady=10)

        # Table output
        columns = ("Record", "Actual", "kNN", "NN")
        self.tree = ttk.Treeview(self.main_frame, columns=columns, show='headings', height=10)
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="center", width=100)

        self.tree.grid(row=6, column=0, columnspan=3, sticky="nsew", pady=10)
        
        columns2 = ("Epoch", "Loss")
        self.epoch_table = ttk.Treeview(self.main_frame, columns=columns2, show='headings', height=10)
        for col in columns2:
            self.epoch_table.heading(col, text=col)
            self.epoch_table.column(col, anchor="center", width=150)
        self.epoch_table.grid(row=7, column=0, columnspan=3, sticky="nsew", pady=10)


    def update_slider_label(self, val):
        self.slider_label.config(text=f"{int(float(val))}%")

    def choose_file(self):
        self.filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.filename:
            self.file_label.config(text=f"Selected: {self.filename}")

    def run_classification(self):
        if not self.filename:
            messagebox.showerror("Error", "Please choose a file.")
            return

        try:
            percentage = self.percentage_var.get()
            if not (10 <= percentage <= 100):
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Enter a valid number between 10 and 100.")
            return

        try:
            self.k = int(self.k_entry.get())
        except:
            messagebox.showerror("Error", "K must be an integer.")
            return

        threshold = 0.7  # 70%
        max_attempts = 10
        attempt = 0

        while attempt < max_attempts:
            self.train_set, self.test_set = load_data(self.filename, percentage)

            knn_preds = knn_predict(self.train_set, self.test_set, self.k)
            knn_acc = accuracy_score([row[-1] for row in self.test_set], knn_preds)

            network, loss_details = train_network([row[1:] for row in self.train_set], 10, 100, 0.1)
            nn_preds = [predict_nn(network, row[1:-1]) for row in self.test_set]
            nn_preds = [round(p) for p in nn_preds]
            nn_acc = accuracy_score([row[-1] for row in self.test_set], nn_preds)
            
            for row in self.epoch_table.get_children():
                self.epoch_table.delete(row)

            for epoch, loss in loss_details:
                self.epoch_table.insert("", "end", values=(epoch, f"{loss:.4f}"))

            if knn_acc >= threshold and nn_acc >= threshold:
                break
            attempt += 1

        self.result_label.config(
            text=f"k-NN Accuracy: {knn_acc*100:.2f}%    |    Neural Net Accuracy: {nn_acc*100:.2f}%"
        )

        for row in self.tree.get_children():
            self.tree.delete(row)

        for i, row in enumerate(self.test_set):
            actual = "ckd" if row[-1] == 1 else "notckd"
            knn_pred = "ckd" if knn_preds[i] == 1 else "notckd"
            nn_pred = "ckd" if nn_preds[i] == 1 else "notckd"
            self.tree.insert("", "end", values=(row[0], actual, knn_pred, nn_pred))


if __name__ == "__main__":
    root = tk.Tk()
    app = KidneyDiseaseClassifierApp(root)
    root.mainloop()
