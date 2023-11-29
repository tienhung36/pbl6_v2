import os
import sys
import threading
import time
import openpyxl
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtGui import QPixmap, QMovie
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QToolBar, QStatusBar, QTableWidget, QPushButton, \
    QFileDialog, QLabel
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib_inline.backend_inline import FigureCanvas
from skimage.transform import rotate, resize, rescale  ## Image rotation routine
from skimage import io, color
import numpy as np
import torch, torch.nn as nn
import cv2
from IPython.display import clear_output
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from skimage.metrics import structural_similarity as ssim  # Import hàm ssim từ thư viện scikit-image
import time


def make_torch_system_matrix(nxd):
    nphi = int(nxd * 1.42)
    nrd = int(nxd * 1.42)
    system_matrix = torch.zeros(nrd * nphi, nxd * nxd)
    for xv in range(nxd):
        for yv in range(nxd):
            for ph in range(nphi):
                yp = -(xv - (nxd * 0.5)) * np.sin(ph * np.pi / nphi) + (yv - (nxd * 0.5)) * np.cos(ph * np.pi / nphi)
                yp_bin = int(yp + nrd / 2.0)
                system_matrix[yp_bin + ph * nrd, xv + yv * nxd] = 1.0
    return system_matrix


def torch_to_np(tarray):
    return np.squeeze(tarray.detach().cpu().numpy())


def np_to_00torch(nparray):
    return torch.from_numpy(nparray).float().unsqueeze(0).unsqueeze(0)


def fp_system_torch(image, sys_mat, nxd, nphi, nrd):
    return torch.reshape(torch.mm(sys_mat, torch.reshape(image, (nxd * nxd, 1))), (nphi, nrd))


def bp_system_torch(sino, sys_mat, nxd, nphi, nrd):
    return torch.reshape(torch.mm(sys_mat.T, torch.reshape(sino, (nrd * nphi, 1))), (nxd, nxd))


def true_sino_object_from_dir(d='img_projection/anh.png', nxd=90):
    nphi = int(nxd * 1.42)
    nrd = int(nxd * 1.42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sys_mat = make_torch_system_matrix(nxd=90).to(device)
    image1 = cv2.imread(d, cv2.IMREAD_GRAYSCALE)
    true_object_np = resize(image1, (nxd, nxd))
    true_object_torch = np_to_00torch(true_object_np).to(device)
    true_sinogram_torch = fp_system_torch(true_object_torch, sys_mat, nxd, nrd, nphi)
    return true_object_torch, true_sinogram_torch, true_object_np


class MLEM_CNN_Net(nn.Module):
    def __init__(self, cnn, sino_for_reconstruction, num_its):
        super(MLEM_CNN_Net, self).__init__()
        nxd = 90
        nphi = int(nxd * 1.42)
        nrd = int(nxd * 1.42)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.sys_mat = make_torch_system_matrix(nxd=90).to(self.device)
        self.num_its = num_its
        self.sino_ones = torch.ones_like(sino_for_reconstruction)
        self.sens_image = bp_system_torch(self.sino_ones, self.sys_mat, nxd, nrd, nphi)
        self.cnn = cnn

    def forward(self, sino_for_reconstruction):
        nxd = 90
        nphi = int(nxd * 1.42)
        nrd = int(nxd * 1.42)
        recon = torch.ones(nxd, nxd).to(self.device)
        for it in range(self.num_its):
            fpsino = fp_system_torch(recon, self.sys_mat, nxd, nrd, nphi)
            ratio = sino_for_reconstruction / (fpsino + 1.0e-9)
            correction = bp_system_torch(ratio, self.sys_mat, nxd, nrd, nphi) / (self.sens_image + 1.0e-15)
            recon = recon * correction
            # lọc hình ảnh MLEM thông qua CNN và thêm lại nó
            cnnrecon = self.cnn(recon)
            recon = torch.abs(recon + cnnrecon)
        return recon, fpsino, ratio, correction, cnnrecon


class UNetCNN(nn.Module):

    def __init__(self, in_channels=1, out_channels=1):
        super(UNetCNN, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Middle (bottleneck)
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = x.unsqueeze(0).unsqueeze(0)  # Add channel dim
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.squeeze(x)  # Remove channel dim
        return x


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowTitle('Editor')
        self.setFixedSize(1030, 800)

        self.start_button = QPushButton('Start', self)
        self.stop_button = QPushButton('Stop', self)
        self.choose_image = QPushButton('Choose Image', self)
        self.noti_text = QLabel('', self)
        self.noti_text.move(10, 40)

        self.start_button.move(10, 10)
        self.stop_button.move(100, 10)
        self.choose_image.move(190, 10)
        self.image_path = ""
        self.choose_image.clicked.connect(self.getFile)
        self.start_button.clicked.connect(self.threading_run)
        self.running_bit = False
        self.stop_button.clicked.connect(self.threading_stop)

        self.image_show = QLabel(self)
        self.image_show.setFixedSize(900, 700)
        self.image_show.move(10, 50)

        # Label Create
        self.label = QLabel(self)
        self.label.setGeometry(QtCore.QRect(25, 25, 50, 50))
        self.label.setMinimumSize(QtCore.QSize(250, 250))
        self.label.setMaximumSize(QtCore.QSize(250, 250))
        self.label.setObjectName("lb1")
        self.label.move(10, 60)

        # Loading the GIF
        self.movie = QMovie("Spinner-1s-42px.gif")
        self.label.setMovie(self.movie)
        self.label.setVisible(False)

        self.startAnimation()


        self.show()


    def startAnimation(self):
        self.movie.start()

        # Stop Animation(According to need)

    def stopAnimation(self):
        self.movie.stop()

    def getFile(self):
        try:
            fname = QFileDialog.getOpenFileName(self, 'Open file',
                                                'c:\\', "Image files (*.jpg *.gif *.png)")
            if fname[0] != "":
                self.image_path = fname[0]
                pixmap = QPixmap(self.image_path)
                self.image_show.clear()
                self.image_show.setPixmap(pixmap)
                self.resize(pixmap.width(), pixmap.height())

        except Exception as ex:
            print(ex)

    def threading_run(self):
        main_thread = threading.Thread(target=(self.main_run), args=())
        main_thread.start()

    def threading_stop(self):
        self.running_bit = True
        self.noti_text.setText("Stopping, Please Wait")

    def addmpl(self, fig):
        self.canvas = FigureCanvas(fig)
        self.mplvl.addWidget(self.canvas)
        self.canvas.draw()
    def main_run(self):
        try:
            self.label.setVisible(True)
            self.startAnimation()
            true_object_torch, true_sinogram_torch, true_object_np = true_sino_object_from_dir(self.image_path)
            print(true_object_torch, true_sinogram_torch, true_object_np)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            cnn = UNetCNN().to(device)
            total_params = sum(p.numel() for p in cnn.parameters())
            print(f"Total Parameters: {total_params}")

            print(cnn)
            # MLEM CNN Net class
            core_iteration = 5

            # Create MLEM CNN Net
            cnnmlem = MLEM_CNN_Net(cnn, true_sinogram_torch, core_iteration).to(device)
            optimiser = torch.optim.Adam(cnnmlem.parameters(), lr=1e-4)
            epochs = 2600

            PSNR_val = list()
            SSIM_val = list()
            train_loss = list()
            epoch_time = list()
            runtime = list()
            loss_fun = nn.MSELoss()
            total_runtime = 0
            ep = 0
            self.running_bit = False
            self.label.setVisible(False)

            while ep < epochs:
                self.noti_text.setText("Epochs " + str(ep))

                start_time = time.time()
                recon, fpsino, ratio, correction, cnnrecon = cnnmlem(true_sinogram_torch)
                loss = loss_fun(recon, torch.squeeze(true_object_torch))

                fbp_recon_np = torch_to_np(recon)
                PSNR_val.append(cv2.PSNR(fbp_recon_np.astype(np.float32) / np.max(fbp_recon_np) * 255,
                                         true_object_np.astype(np.float32) / np.max(true_object_np) * 255))
                SSIM_val.append(ssim(fbp_recon_np, true_object_np, data_range=1))

                train_loss.append(loss.item())
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()

                end_time = time.time()
                epoch_runtime = end_time - start_time
                epoch_time.append(epoch_runtime)
                total_runtime += epoch_runtime
                runtime.append(total_runtime)

                if self.running_bit:
                    self.noti_text.setText("Stopped")

                    break

                if ep % 1 == 0:
                    img1 = cv2.resize(255 * torch_to_np(recon), [369, 369])
                    cv2.imwrite("hello_world.png", img1)
                    time.sleep(1)
                    pixmap = QPixmap("hello_world.png")
                    self.image_show.clear()
                    self.image_show.setPixmap(pixmap)
                    self.resize(pixmap.width(), pixmap.height())
                ep += 1
        except Exception as ex:
            print(ex)


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())
