import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import sys
import os
from pathlib import Path
from validate_employees import EasyOCRProcessor, validate_document, normalize_arabic_text, post_process_arabic_text, fuzzy_search_arabic
import pandas as pd
from datetime import datetime

class ValidationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Employee Document Validator")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        self.excel_file = None
        self.test_folder = None
        self.output_file = None
        self.ocr_processor = None
        self.is_running = False
        
        self.setup_ui()
        
    def setup_ui(self):
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill='x', padx=0, pady=0)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, 
            text="Employee Document Validator",
            font=('Arial', 16, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(pady=15)
        
        input_frame = tk.LabelFrame(self.root, text="Input Files", font=('Arial', 11, 'bold'), bg='#f0f0f0', padx=10, pady=10)
        input_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(input_frame, text="Excel File:", bg='#f0f0f0', font=('Arial', 10), anchor='w').grid(row=0, column=0, sticky='w', pady=5, padx=(0, 10))
        self.excel_entry = tk.Entry(input_frame, width=50, font=('Arial', 9))
        self.excel_entry.grid(row=0, column=1, padx=10, pady=5)
        tk.Button(input_frame, text="Browse", command=self.browse_excel, bg='#3498db', fg='white', font=('Arial', 9)).grid(row=0, column=2, pady=5)
        
        tk.Label(input_frame, text="Documents Folder:", bg='#f0f0f0', font=('Arial', 10), anchor='w').grid(row=1, column=0, sticky='w', pady=5, padx=(0, 10))
        self.folder_entry = tk.Entry(input_frame, width=50, font=('Arial', 9))
        self.folder_entry.grid(row=1, column=1, padx=10, pady=5)
        tk.Button(input_frame, text="Browse", command=self.browse_folder, bg='#3498db', fg='white', font=('Arial', 9)).grid(row=1, column=2, pady=5)
        
        tk.Label(input_frame, text="Output File:", bg='#f0f0f0', font=('Arial', 10), anchor='w').grid(row=2, column=0, sticky='w', pady=5, padx=(0, 10))
        self.output_entry = tk.Entry(input_frame, width=50, font=('Arial', 9))
        self.output_entry.grid(row=2, column=1, padx=10, pady=5)
        tk.Button(input_frame, text="Browse", command=self.browse_output, bg='#3498db', fg='white', font=('Arial', 9)).grid(row=2, column=2, pady=5)
        
        control_frame = tk.Frame(self.root, bg='#f0f0f0')
        control_frame.pack(fill='x', padx=20, pady=10)
        
        self.start_btn = tk.Button(
            control_frame, 
            text="Start Validation",
            command=self.start_validation,
            bg='#27ae60',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=20,
            pady=10
        )
        self.start_btn.pack(side='left', padx=5)
        
        self.stop_btn = tk.Button(
            control_frame,
            text="Stop",
            command=self.stop_validation,
            bg='#e74c3c',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=20,
            pady=10,
            state='disabled'
        )
        self.stop_btn.pack(side='left', padx=5)
        
        progress_frame = tk.LabelFrame(self.root, text="Progress", font=('Arial', 11, 'bold'), bg='#f0f0f0', padx=10, pady=10)
        progress_frame.pack(fill='x', padx=20, pady=10)
        
        self.progress = ttk.Progressbar(progress_frame, length=800, mode='determinate')
        self.progress.pack(fill='x', pady=5)
        
        self.status_label = tk.Label(progress_frame, text="Ready", font=('Arial', 10), bg='#f0f0f0')
        self.status_label.pack(pady=5)
        
        stats_frame = tk.Frame(progress_frame, bg='#f0f0f0')
        stats_frame.pack(fill='x', pady=5)
        
        self.valid_label = tk.Label(stats_frame, text="Valid: 0", font=('Arial', 10, 'bold'), bg='#f0f0f0', fg='#27ae60')
        self.valid_label.pack(side='left', padx=20)
        
        self.invalid_label = tk.Label(stats_frame, text="Invalid: 0", font=('Arial', 10, 'bold'), bg='#f0f0f0', fg='#e74c3c')
        self.invalid_label.pack(side='left', padx=20)
        
        self.error_label = tk.Label(stats_frame, text="Errors: 0", font=('Arial', 10, 'bold'), bg='#f0f0f0', fg='#f39c12')
        self.error_label.pack(side='left', padx=20)
        
        log_frame = tk.LabelFrame(self.root, text="Log", font=('Arial', 11, 'bold'), bg='#f0f0f0', padx=10, pady=10)
        log_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, font=('Courier', 9), bg='#2c3e50', fg='#ecf0f1', wrap='word')
        self.log_text.pack(fill='both', expand=True)
        
    def browse_excel(self):
        filename = filedialog.askopenfilename(
            title="Select Excel File",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        if filename:
            self.excel_entry.delete(0, tk.END)
            self.excel_entry.insert(0, filename)
            default_output = os.path.join(
                os.path.dirname(filename),
                f"employees_validated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            )
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, default_output)
            
    def browse_folder(self):
        foldername = filedialog.askdirectory(title="Select Documents Folder")
        if foldername:
            self.folder_entry.delete(0, tk.END)
            self.folder_entry.insert(0, foldername)
            
    def browse_output(self):
        filename = filedialog.asksaveasfilename(
            title="Save Output File",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if filename:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, filename)
            
    def log(self, message):
        self.log_text.insert(tk.END, message + '\n')
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def start_validation(self):
        self.excel_file = self.excel_entry.get()
        self.test_folder = self.folder_entry.get()
        self.output_file = self.output_entry.get()
        
        if not self.excel_file or not os.path.exists(self.excel_file):
            messagebox.showerror("Error", "Please select a valid Excel file")
            return
            
        if not self.test_folder or not os.path.exists(self.test_folder):
            messagebox.showerror("Error", "Please select a valid documents folder")
            return
            
        if not self.output_file:
            messagebox.showerror("Error", "Please specify an output file")
            return
            
        self.is_running = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        
        self.log_text.delete(1.0, tk.END)
        self.log("=" * 80)
        self.log("EMPLOYEE DOCUMENT VALIDATION")
        self.log("=" * 80)
        
        thread = threading.Thread(target=self.run_validation, daemon=True)
        thread.start()
        
    def stop_validation(self):
        self.is_running = False
        self.log("\nValidation stopped by user")
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        
    def run_validation(self):
        try:
            self.log(f"\nReading Excel file: {self.excel_file}")
            df = pd.read_excel(self.excel_file)
            
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            
            self.log(f"Total employees: {len(df)}")
            self.log(f"Documents folder: {self.test_folder}")
            
            document_column = 'بيان حالة وظيفية'
            validation_column = 'صحة بيان الحالة'
            
            if document_column not in df.columns:
                self.log(f"\nError: Column '{document_column}' not found")
                self.log(f"Available columns: {df.columns.tolist()}")
                messagebox.showerror("Error", f"Column '{document_column}' not found in Excel file")
                self.start_btn.config(state='normal')
                self.stop_btn.config(state='disabled')
                return
                
            self.log("\nInitializing OCR processor...")
            self.status_label.config(text="Initializing OCR...")
            
            import cv2
            import numpy as np
            self.ocr_processor = EasyOCRProcessor()
            
            self.log("OCR processor ready!")
            
            validation_results = []
            valid_count = 0
            invalid_count = 0
            error_count = 0
            
            self.progress['maximum'] = len(df)
            self.progress['value'] = 0
            
            self.log("\nStarting validation...")
            self.log("-" * 80)
            
            for idx, row in df.iterrows():
                if not self.is_running:
                    break
                    
                employee_name = row.get('الاسم رباعي', f'Employee_{idx}')
                document_filename = row.get(document_column)
                
                self.log(f"\n[{idx + 1}/{len(df)}] Processing: {employee_name}")
                self.status_label.config(text=f"Processing {idx + 1}/{len(df)}: {employee_name}")
                
                if pd.isna(document_filename) or not document_filename:
                    self.log(f"  No document specified")
                    validation_results.append(0)
                    error_count += 1
                    self.error_label.config(text=f"Errors: {error_count}")
                    self.progress['value'] = idx + 1
                    continue
                    
                file_path = os.path.join(self.test_folder, document_filename)
                
                if not os.path.exists(file_path):
                    self.log(f"  File not found: {document_filename}")
                    validation_results.append(0)
                    error_count += 1
                    self.error_label.config(text=f"Errors: {error_count}")
                    self.progress['value'] = idx + 1
                    continue
                    
                self.log(f"  Validating: {document_filename}")
                validation_result = validate_document(file_path, self.ocr_processor, verbose=False)
                
                if validation_result['success']:
                    self.log(f"  VALID")
                    validation_results.append(1)
                    valid_count += 1
                    self.valid_label.config(text=f"Valid: {valid_count}")
                else:
                    self.log(f"  INVALID: {validation_result['error']}")
                    validation_results.append(0)
                    invalid_count += 1
                    self.invalid_label.config(text=f"Invalid: {invalid_count}")
                    
                self.progress['value'] = idx + 1
                
            if self.is_running:
                df[validation_column] = validation_results
                
                self.log("\nSaving results...")
                self.status_label.config(text="Saving results...")
                df.to_excel(self.output_file, index=False, engine='openpyxl')
                
                self.log("\n" + "=" * 80)
                self.log("VALIDATION COMPLETE")
                self.log("=" * 80)
                self.log(f"Valid documents: {valid_count}")
                self.log(f"Invalid documents: {invalid_count}")
                self.log(f"Errors/Missing: {error_count}")
                self.log(f"Total processed: {len(df)}")
                self.log(f"\nUpdated Excel saved to: {self.output_file}")
                self.log("=" * 80)
                
                self.status_label.config(text="Validation Complete!")
                messagebox.showinfo("Success", f"Validation complete!\n\nValid: {valid_count}\nInvalid: {invalid_count}\nErrors: {error_count}\n\nOutput saved to:\n{self.output_file}")
            
        except Exception as e:
            self.log(f"\nERROR: {str(e)}")
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
            
        finally:
            self.start_btn.config(state='normal')
            self.stop_btn.config(state='disabled')
            self.is_running = False

def main():
    root = tk.Tk()
    app = ValidationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
