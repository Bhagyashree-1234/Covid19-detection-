from tkinter import *
import tkinter.messagebox
from PIL import ImageTk, Image
import cv2
from tkinter import filedialog
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np

img_path = ""

def fileselector():
    global img_path
    main_win = Tk() 
    main_win.withdraw()

    main_win.overrideredirect(True)
    main_win.geometry('0x0+0+0')

    main_win.deiconify()
    main_win.lift()
    main_win.focus_force()

    main_win.sourceFile = filedialog.askopenfilename(filetypes=(("Image Files", ("*.jpg", "*.png", "*.jpeg")), ("All Files", "*")), parent=main_win, initialdir="./Testing",
                                                      title='Please select a X-Ray Image')
    main_win.destroy()

    img_path = main_win.sourceFile
    print(img_path)
    tkinter.messagebox.showinfo("Image Selected", "Click on Detect Button.\nTo get the COVID Prediction")


def predict():
    global result_var
    if img_path == "":
        tkinter.messagebox.showinfo("Image Not Selected", "Please Select X-Ray Image\nTo get the COVID Prediction")
    else:
        print("[INFO] loading network...")
        model = load_model('./covid_pypower.h5')

        labels = ['Covid', 'Normal']  
        start_point = (15, 15)
        end_point = (230, 80) 
        thickness = -1
        
        print("[INFO] reading image...")
        frame = cv2.imread(img_path)

        roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.resize(frame, (224, 224))
        roi = roi_gray.astype('float') / 255.0 
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        print("[INFO] classifying image...")

        preds = model.predict(roi)[0]
        label = labels[preds.argmax()]

        if label == 'Covid':
            image = cv2.rectangle(frame, start_point, end_point, (0, 0, 255), thickness)
            cv2.putText(image, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        else:
            image = cv2.rectangle(frame, start_point, end_point, (0, 255, 0), thickness)
            cv2.putText(image, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)

        cv2.imshow('COVID Detector', frame)

        print("[INFO] saving image...")
        cv2.imwrite("./Output/detected13.jpg", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if label == 'Covid':
            tkinter.messagebox.showinfo("COVID Predicted!", "Take Care. Be Alert.\t \nEat Healthy\n Stay Safe.")
        else:
            tkinter.messagebox.showinfo("NORMAL Report!", "Don't Worry! \nYour Report is Normal\n Stay Safe")


root = Tk()
root.title("GUI : COVID Detection")

# Apply styles
root.configure(background='black')

# Get screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

Tops = Frame(root, bg='black', pady=1, width=screen_width, height=screen_height, relief="ridge")
Tops.pack(expand=True, fill=BOTH)

Title_Label = Label(Tops, font=('Comic Sans MS', 20, 'bold'),
                    text="Bhagyashree Presents 'Coronavirus Detection' \nUsing Artificial Intelligence", pady=9,
                    bg='black', fg='pink', justify="center")
Title_Label.pack(expand=True, fill=BOTH)

MainFrame = Frame(root, bg='black', pady=2, padx=2, width=screen_width, height=screen_height, relief=RIDGE)
MainFrame.pack(expand=True, fill=BOTH)

Label_1 = Label(MainFrame, font=('lato black', 17, 'bold'), text="\tCOVID Prediction using Chest X-Ray", padx=2, pady=2,
                bg="black", fg="white", justify="center")
Label_1.pack(expand=True, fill=BOTH)

Label_2 = Label(MainFrame, font=('arial', 15, 'bold'), text="", padx=2, pady=2, bg="black", fg="white")
Label_2.pack(expand=True, fill=BOTH)

# Create a frame to contain buttons
button_frame = Frame(MainFrame, bg='black')
button_frame.pack(expand=True, fill=BOTH)

Label_9 = Button(button_frame, text="Select X-ray Image", width=20, height=2, bg='red', fg="black", font=('Arial', 14, 'bold'), command=fileselector)
Label_9.pack(side=LEFT, padx=50, pady=10)

Label_10 = Button(button_frame, text="Detect Coronavirus", width=20, height=2, bg='red', fg="black", font=('Arial', 14, 'bold'), command=predict)
Label_10.pack(side=RIGHT, padx=50, pady=10)

Label_2 = Label(MainFrame, font=('arial', 10, 'bold'), text="", padx=2, pady=2, bg="black", fg="white")
Label_2.pack(expand=True, fill=BOTH)

Label_3 = Label(MainFrame, font=('arial', 30, 'bold'), text="\t\t\t\t          ", padx=2, pady=2, bg="black",
                fg="white")
Label_3.pack(expand=True, fill=BOTH)

# Resize and display images
img = cv2.imread("./covid19.png")
img = cv2.resize(img, (500, 350))
cv2.imwrite('covid19.png', img)
img = ImageTk.PhotoImage(Image.open("covid19.png"))
panel = Label(MainFrame, image=img, bg='black')  # Set background color for image panel
panel.pack(expand=True, fill=BOTH)

# Create a frame to display prediction result
result_frame = Frame(root, bg='black', padx=10, pady=10)
result_frame.pack(side=TOP, anchor=NW)

result_label = Label(result_frame, font=('Arial', 14, 'bold'), bg='black', fg='white', text="COVID Prediction Result:")
result_label.pack(side=LEFT)

result_var = StringVar()
result_var.set("No prediction yet.")
result_display = Label(result_frame, font=('Arial', 14), bg='black', fg='white', textvariable=result_var)
result_display.pack(side=LEFT)

root.mainloop()






