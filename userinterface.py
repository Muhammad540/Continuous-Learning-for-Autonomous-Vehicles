import os
from tkinter import Tk, Label, Entry, Button, messagebox
from tkinter import ttk
from PIL import Image, ImageTk

def save_labeled_image(image, label, folder="labelled_images"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_path = os.path.join(folder, f"{label}.png")
    image.save(file_path)

def display_and_label_image(array):
    image = Image.fromarray(array)

    # Initialize Tkinter root window
    root = Tk()
    root.title("Image Annotation Tool")

    # Calculate window size based on image size
    img_width, img_height = image.size
    window_width = img_width + 40
    window_height = img_height + 150
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width/2 - window_width/2)
    center_y = int(screen_height/2 - window_height/2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

    # Styling
    style = ttk.Style(root)
    style.configure('TButton', font=('Helvetica', 10, 'bold'))
    style.configure('TLabel', font=('Helvetica', 12), background='#f0f0f0')
    style.configure('TEntry', font=('Helvetica', 10))
    root.configure(bg='#f0f0f0')

    # Header label with larger, attractive font and red color
    header_label = Label(root, text="Novelty Detected! Please Classify it", font=('Arial', 18, 'bold'), fg='red', bg='#f0f0f0')
    header_label.grid(row=0, column=0, columnspan=2, pady=(10, 20))

    # Displaying the image in its original size
    tk_image = ImageTk.PhotoImage(image)
    img_label = Label(root, image=tk_image, borderwidth=2, relief="solid")
    img_label.image = tk_image
    img_label.grid(row=1, column=0, columnspan=2, padx=20, pady=20)

    # Label entry
    label_entry = ttk.Entry(root, font=('Helvetica', 10))
    label_entry.grid(row=2, column=1, padx=20, pady=(0, 10))

    label_entry_label = ttk.Label(root, text="Enter a label for the image:", font=('Helvetica', 10), background='#f0f0f0')
    label_entry_label.grid(row=2, column=0, sticky='w', padx=20, pady=(0, 10))

    # Button for saving label
    def on_button_click():
        label = label_entry.get()
        if label:
            save_labeled_image(image, label)
            messagebox.showinfo("Success", "Label saved successfully!")
            root.destroy()
        else:
            messagebox.showerror("Error", "Please enter a label")

    button = ttk.Button(root, text="Save Label", command=on_button_click)
    button.grid(row=3, column=0, columnspan=2, pady=10)

    # Add padding to all children of root
    for child in root.winfo_children():
        child.grid_configure(padx=5, pady=5)

    root.mainloop()

