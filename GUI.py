from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import ip_exp
from timeit import default_timer as timer
import numpy as np

def get_btns():
    g_btn = Button(root, text='Gray Scale', command=grey_scale,padx=5,pady=5,bd=4)
    g_btn.pack()
    g_btn.place(relx=0.05,rely=0.8)

    g_btn = Button(root, text='Negative', command=negative_image,padx=5,pady=5,bd=4)
    g_btn.pack()
    g_btn.place(relx=0.1,rely=0.8)

    g_btn = Button(root, text='Thresholding', command=threshold_image,padx=5,pady=5,bd=4)
    g_btn.pack()
    g_btn.place(relx=0.19,rely=0.8)
   
    g_btn = Button(root, text='Gray Level Slicing', command=graylevelslicing,padx=5,pady=5,bd=4)
    g_btn.pack()
    g_btn.place(relx=0.25,rely=0.8)

    g_btn = Button(root, text='BitPlane Slicing', command=bitplaneslicing,padx=5,pady=5,bd=4)
    g_btn.pack()
    g_btn.place(relx=0.32,rely=0.8)

    g_btn = Button(root, text='Contrast streaching', command=contraststreaching,padx=5,pady=5,bd=4)
    g_btn.pack()
    g_btn.place(relx=0.05,rely=0.9)

    g_btn = Button(root, text='Histogram', command=histogram,padx=5,pady=5,bd=4)
    g_btn.pack()
    g_btn.place(relx=0.13,rely=0.9)

    g_btn = Button(root, text='Box-s', command=box_image,padx=5,pady=5,bd=4)
    g_btn.pack()
    g_btn.place(relx=0.5,rely=0.8)

    g_btn = Button(root, text='Median-s', command=median_image,padx=5,pady=5,bd=4)
    g_btn.pack()
    g_btn.place(relx=0.54,rely=0.8)

    g_btn = Button(root, text='W_avg-s', command=w_avg_image,padx=5,pady=5,bd=4)
    g_btn.pack()
    g_btn.place(relx=0.59,rely=0.8)

    g_btn = Button(root, text='Robert', command=Robert_image,padx=5,pady=5,bd=4)
    g_btn.pack()
    g_btn.place(relx=0.5,rely=0.9)

    g_btn = Button(root, text='Prewitt', command=Prewitt_image,padx=5,pady=5,bd=4)
    g_btn.pack()
    g_btn.place(relx=0.54,rely=0.9)
    
    g_btn = Button(root, text='Sobel', command=Sobel_image,padx=5,pady=5,bd=4)
    g_btn.pack()
    g_btn.place(relx=0.58,rely=0.9)

    g_btn = Button(root, text='Laplacian', command=Laplacian_image,padx=5,pady=5,bd=4)
    g_btn.pack()
    g_btn.place(relx=0.62,rely=0.9)
   
    g_btn = Button(root, text='Canny_Edge', command=Cannyedge_image,padx=5,pady=5,bd=4)
    g_btn.pack()
    g_btn.place(relx=0.75,rely=0.9)
   
    
    

def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename

def open_img():
    global orignal_img
    x = openfn()
    orignal_img = Image.open(x)
    orignal_img = orignal_img.resize((800, 480), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(orignal_img)
    panel = Label(root, image=img,relief=SUNKEN,bd=10,height=480,width=800)
    panel.image = img
    panel.pack()
    panel.place(relx=0.005, rely=0.2)
    text=Label(root,text='>>>>')
    text.pack()
    text.place(relx=0.49,rely=0.4)
    get_btns()
    
def new_img_open(new_img,flag=0):
    if flag==0:
        new_img = Image.fromarray(new_img)
    new_img = ImageTk.PhotoImage(new_img)
    panel = Label(root, image=new_img,relief=SUNKEN,bd=10,height=480,width=800)
    panel.image = new_img
    panel.pack()
    panel.place(relx=0.52, rely=0.2)

def input_window():
    input_w = Toplevel()
    input_w.title("Input Values")
    input_w.geometry("200x200")
    input_w.resizable(width=True, height=True)
    Label(input_w,text="Enter A :").grid(row=0)
    E1=Entry(input_w,textvariable=A,width=10)
    Label(input_w,text="Enter B :").grid(row=1)
    E2=Entry(input_w,textvariable=B,width=10)
    E1.grid(row=0,column=1)
    E2.grid(row=1,column=1)
    submit=Button(input_w, text='Submit', command=lambda:flag.set(1),padx=5,pady=5,bd=4)
    submit.grid(row=3,column=0)
    submit.wait_variable(flag)
    a=int(A.get())
    b=int(B.get())
    input_w.destroy()
    return a,b

def multi_image_window(planes):
    mw=Toplevel()
    mw.title=("Bit plane images")
    mw.geometry("800x600")
    count=0
    for i in range(len(planes)-1,-1,-1):
        new_img=planes[i]
        new_img = Image.fromarray(new_img.astype(np.uint8))
        new_img = new_img.resize((400, 400), Image.ANTIALIAS)
        new_img = ImageTk.PhotoImage(new_img)
        panel = Label(mw, image=new_img,relief=SUNKEN,bd=4,height=400,width=400)
        panel.image = new_img
        panel.pack()
        if count<4:
            panel.place(relx=0.1*(count)*2.5,rely=0.05)
            count+=1
        else:
            panel.place(relx=0.1*(count-4)*2.5,rely=0.55)
            count+=1
    pass

def grey_scale():
    start = timer()
    new_img = ip_exp.grayscale(np.array(orignal_img))
    print('Execution time :',timer()-start)
    new_img_open(new_img)
   

def negative_image():
    start = timer()
    new_img = ip_exp.negative_img(np.array(orignal_img))
    print('Execution time :',timer()-start)
    new_img_open(new_img)

def graylevelslicing():
    a,b=input_window()
    if a==0 and b==0:
        a,b=80,150
    start = timer()
    new_img = ip_exp.gray_level_slicing(np.array(orignal_img),a,b)
    print('Execution time :',timer()-start)
    new_img_open(new_img)

def threshold_image():
    a,b=input_window()
    if a==0 and b==0:
        a,b=80,150
    start = timer()
    new_img = ip_exp.gray_level_thresholding(np.array(orignal_img),a,b)
    print('Execution time :',timer()-start)
    new_img_open(new_img)

def bitplaneslicing():
    start = timer()
    planes=[]
    planes = ip_exp.bit_plane_splicing(np.array(orignal_img))
    print('Execution time :',timer()-start)
    multi_image_window(planes)

def contraststreaching():
    s1,s2=input_window()
    if s2==0:
        s2=255
    start = timer()
    new_img = ip_exp.contrast_streaching(np.array(orignal_img),s1,s2)
    print('Execution time :',timer()-start)
    new_img_open(new_img)


def histogram():
    start = timer()
    new_img= ip_exp.hist(np.array(orignal_img))
    print('Execution time :', timer()-start)
    new_img_open(new_img,1)


def box_image():
    start = timer()
    new_img = ip_exp.box_filter(orignal_img)
    print('Execution time :', timer()-start)
    new_img_open(new_img,1)

def median_image():
    start = timer()
    new_img = ip_exp.median_filter(orignal_img)
    print('Execution time :', timer()-start)
    new_img_open(new_img,1)

def w_avg_image():
    start = timer()
    new_img = ip_exp.weighted_filter(orignal_img)
    print('Execution time :', timer()-start)
    new_img_open(new_img,1)


def Sobel_image():
    start = timer()
    new_img = ip_exp.Sobel_filter(orignal_img)
    print('Execution time :', timer()-start)
    new_img_open(new_img,1)

def Prewitt_image():
    start = timer()
    new_img = ip_exp.Prewitt_filter(orignal_img)
    print('Execution time :', timer()-start)
    new_img_open(new_img,1)

def Robert_image():
    start = timer()
    new_img = ip_exp.Robert_filter(orignal_img)
    print('Execution time :', timer()-start)
    new_img_open(new_img,1)

def Cannyedge_image():
    start = timer()
    new_img = ip_exp.canny_main(orignal_img)
    print('Execution time :', timer()-start)
    new_img_open(new_img,1)


def Laplacian_image():
    start = timer()
    new_img = ip_exp.laplacian_sharpening(orignal_img)
    new_img = np.asarray(orignal_img) - np.asarray(new_img)
    new_img = Image.fromarray(new_img)
    print('Execution time :', timer()-start)
    new_img_open((new_img), 1)

root = Tk()
root.geometry("1000x800")
root.title("Image Processing")
root.resizable(width=True, height=True)
flag = IntVar()
A = IntVar()
B = IntVar()

btn = Button(root, text='open image', command=open_img,padx=5,pady=5,bd=4)
btn.pack()
btn.place(relx=0.5,rely=0.05)

root.mainloop()