import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess
import os
import threading

class ImageMatchingGUI:
    MATCHING_TYPES = {
        "Baseline Matching": "1",
        "Histogram Matching": "2",
        "Multi-histogram Matching": "3",
        "Texture and Color Matching": "4",
        "ResNet Features Matching": "5",
        "Custom Feature Matching": "6"
    }
    
    EXECUTABLE_NAME = os.path.join("x64", "Debug", "Project2.exe")
    
    def __init__(self, root):
        self.root = root
        self.root.title("Image Matching System")
        self.root.geometry("700x600")
        
        self.build_process = None
        self.match_process = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the entire UI"""
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Build tab
        build_tab = ttk.Frame(notebook)
        notebook.add(build_tab, text='Build Features')
        self._setup_build_tab(build_tab)
        
        # Match tab
        match_tab = ttk.Frame(notebook)
        notebook.add(match_tab, text='Match Images')
        self._setup_match_tab(match_tab)
    
    def _setup_build_tab(self, parent):
        """Setup the Build Features tab"""
        frame = ttk.Frame(parent, padding="20")
        frame.pack(fill='both', expand=True)
        
        self.build_dir_var = self._create_field_row(frame, 0, "Image Directory:", 
                                                     self._browse_directory)
        self.build_csv_var = self._create_field_row(frame, 1, "Output CSV File:", 
                                                     self._browse_save_csv)
        self.build_match_var = self._create_combobox_row(frame, 2, "Matching Type:")
        
        ttk.Button(frame, text="Build Feature Database", 
                  command=self._run_build_features, 
                  style='Accent.TButton').grid(row=3, column=1, pady=20)
        
        self.build_output = self._create_output_console(frame, 4)
        
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(5, weight=1)
    
    def _setup_match_tab(self, parent):
        """Setup the Match Images tab"""
        frame = ttk.Frame(parent, padding="20")
        frame.pack(fill='both', expand=True)
        
        self.match_img_var = self._create_field_row(frame, 0, "Target Image:", 
                                                     self._browse_target_image)
        self.match_csv_var = self._create_field_row(frame, 1, "Feature CSV File:", 
                                                     self._browse_open_csv)
        self.match_n_var = self._create_field_row(frame, 2, "Number of Matches:", 
                                                   None, default="5")
        self.match_type_var = self._create_combobox_row(frame, 3, "Matching Type:")
        
        ttk.Button(frame, text="Find Matching Images", 
                  command=self._run_match_images, 
                  style='Accent.TButton').grid(row=4, column=1, pady=20)
        
        self.match_output = self._create_output_console(frame, 5)
        
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(6, weight=1)
    
    def _create_field_row(self, parent, row, label, browse_cmd, default=""):
        """Create a labeled input field with optional browse button"""
        ttk.Label(parent, text=label, font=('Arial', 10, 'bold')).grid(
            row=row, column=0, sticky='w', pady=5)
        
        var = tk.StringVar(value=default)
        ttk.Entry(parent, textvariable=var, width=50).grid(
            row=row, column=1, padx=5, pady=5)
        
        if browse_cmd:
            ttk.Button(parent, text="Browse", command=browse_cmd).grid(
                row=row, column=2, padx=5, pady=5)
        
        return var
    
    def _create_combobox_row(self, parent, row, label):
        """Create a labeled combobox for matching types"""
        ttk.Label(parent, text=label, font=('Arial', 10, 'bold')).grid(
            row=row, column=0, sticky='w', pady=5)
        
        var = tk.StringVar()
        combo = ttk.Combobox(parent, textvariable=var, 
                            values=list(self.MATCHING_TYPES.keys()), 
                            state='readonly', width=47)
        combo.grid(row=row, column=1, padx=5, pady=5)
        combo.current(0)
        
        return var
    
    def _create_output_console(self, parent, start_row):
        """Create output console with label"""
        ttk.Label(parent, text="Output:", font=('Arial', 10, 'bold')).grid(
            row=start_row, column=0, sticky='nw', pady=5)
        
        output = scrolledtext.ScrolledText(parent, height=15, width=70)
        output.grid(row=start_row + 1, column=0, columnspan=3, pady=5, sticky='nsew')
        
        return output
    
    def _browse_directory(self):
        """Browse for image directory"""
        directory = filedialog.askdirectory(title="Select Image Directory")
        if directory:
            self.build_dir_var.set(directory)
    
    def _browse_save_csv(self):
        """Browse for CSV file to save"""
        filename = filedialog.asksaveasfilename(
            title="Save Feature CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.build_csv_var.set(filename)
    
    def _browse_target_image(self):
        """Browse for target image"""
        filename = filedialog.askopenfilename(
            title="Select Target Image",
            filetypes=[("Image files", "*.jpg *.png *.ppm *.tif"), ("All files", "*.*")]
        )
        if filename:
            self.match_img_var.set(filename)
    
    def _browse_open_csv(self):
        """Browse for CSV file to open"""
        filename = filedialog.askopenfilename(
            title="Select Feature CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.match_csv_var.set(filename)
    
    def _run_build_features(self):
        """Run buildFeatures executable"""
        img_dir = self.build_dir_var.get()
        csv_file = self.build_csv_var.get()
        match_type = self.MATCHING_TYPES[self.build_match_var.get()]
        
        if not img_dir or not csv_file:
            messagebox.showerror("Please fill in all fields!")
            return
        
        if not os.path.isdir(img_dir):
            messagebox.showerror("Invalid image directory!")
            return
        
        self._write_to_console(self.build_output, "Building features...\n\n")
        
        cmd = ["build", img_dir, csv_file, match_type]
        thread = threading.Thread(target=self._execute_command, 
                                 args=(cmd, self.build_output, "build"))
        thread.daemon = True
        thread.start()
    
    def _run_match_images(self):
        """Run matchImages executable"""
        target_img = self.match_img_var.get()
        csv_file = self.match_csv_var.get()
        n_matches = self.match_n_var.get()
        match_type = self.MATCHING_TYPES[self.match_type_var.get()]
        
        if not target_img or not csv_file or not n_matches:
            messagebox.showerror("Please fill in all fields!")
            return
        
        if not os.path.isfile(target_img):
            messagebox.showerror("Invalid target image!")
            return
        
        if not os.path.isfile(csv_file):
            messagebox.showerror("Invalid CSV file!")
            return
        
        try:
            if int(n_matches) <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Must be a positive integer!")
            return
        
        self._write_to_console(self.match_output, "Finding matching images...\n\n")
        
        cmd = ["match", target_img, csv_file, n_matches, match_type]
        thread = threading.Thread(target=self._execute_command, 
                                 args=(cmd, self.match_output, "match"))
        thread.daemon = True
        thread.start()
    
    def _execute_command(self, cmd, output_widget, process_type):
        """Execute command in separate thread"""
        try:
            # Terminate previous process
            process = self.build_process if process_type == "build" else self.match_process
            if process and process.poll() is None:
                process.terminate()
                process.wait()
            
            # Build full command
            script_dir = os.path.dirname(os.path.abspath(__file__))
            executable = os.path.join(script_dir, self.EXECUTABLE_NAME)
            full_cmd = [executable] + cmd
            
            # Start process
            process = subprocess.Popen(
                full_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if process_type == "build":
                self.build_process = process
            else:
                self.match_process = process
            
            # Read output in real-time
            for line in process.stdout:
                self._write_to_console(output_widget, line)
            
            process.wait()
            
            # Read any stderr output and show only if there's actual error content
            stderr = process.stderr.read().strip()
            if stderr:
                self._write_to_console(output_widget, f"\nError:\n{stderr}\n")
        
        except FileNotFoundError:
            self._write_to_console(output_widget, 
                f"Error: {self.EXECUTABLE_NAME} not found!\n")
        except Exception as e:
            self._write_to_console(output_widget, f"Error: {str(e)}\n")
    
    def _write_to_console(self, widget, text):
        """Write text to output console and scroll to end"""
        widget.insert(tk.END, text)
        widget.see(tk.END)
        self.root.update()

def main():
    root = tk.Tk()
    app = ImageMatchingGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()