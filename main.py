from enum import Flag
import os
import subprocess
import sys
from os import path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.ExifTags
import PIL.Image
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox
from PyQt5.uic import loadUi
from tqdm import *

terminationCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#################################################################
### Methods for writing/reading calibration files ###
def saveCameraCoef(K, D, rms, sPath):
    """ Save camera matrix and the distortion coefficients to given path - yml file """

    oFile = cv2.FileStorage(sPath, cv2.FILE_STORAGE_WRITE)

    oFile.write("K", K)
    oFile.write("D", D)
    oFile.write("RMS", rms)

    oFile.release()


def saveStereoCoef(K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q, sPath):
    """ Save the stereo coefficients to given path/file """

    oFile = cv2.FileStorage(sPath, cv2.FILE_STORAGE_WRITE)
    oFile.write("K1", K1)
    oFile.write("D1", D1)
    oFile.write("K2", K2)
    oFile.write("D2", D2)
    oFile.write("R", R)
    oFile.write("T", T)
    oFile.write("E", E)
    oFile.write("F", F)
    oFile.write("R1", R1)
    oFile.write("R2", R2)
    oFile.write("P1", P1)
    oFile.write("P2", P2)
    oFile.write("Q", Q)
    oFile.release()


def loadCameraCoef(sPath):
    """ Loads camera matrix and distortion coefficients for single camera calibration """

    oFile = cv2.FileStorage(sPath, cv2.FILE_STORAGE_READ)

    #specify node and type of object 
    K = oFile.getNode("K").mat()
    D = oFile.getNode("D").mat()

    oFile.release()
    return [K, D]

def loadQ(sPath):
    """ Loads camera matrix and distortion coefficients for single camera calibration """

    oFile = cv2.FileStorage(sPath, cv2.FILE_STORAGE_READ)

    #specify node and type of object 
    Q = oFile.getNode("Q").mat()

    oFile.release()
    return Q


def loadStereoCoef(sPath):
    """ Loads coefficients for stereo calibration"""

    oFile = cv2.FileStorage(sPath, cv2.FILE_STORAGE_READ)

    K1 = oFile.getNode("K1").mat()
    D1 = oFile.getNode("D1").mat()
    K2 = oFile.getNode("K2").mat()
    D2 = oFile.getNode("D2").mat()
    R = oFile.getNode("R").mat()
    T = oFile.getNode("T").mat()
    E = oFile.getNode("E").mat()
    F = oFile.getNode("F").mat()
    R1 = oFile.getNode("R1").mat()
    R2 = oFile.getNode("R2").mat()
    P1 = oFile.getNode("P1").mat()
    P2 = oFile.getNode("P2").mat()
    Q = oFile.getNode("Q").mat()

    oFile.release()
    return [K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q]

############################################################################
### Methods for writing/opening PLY files with MeshLab ###
def openMeshLab(sPath):
    """ Open Mesh Lab with given file path"""
    subprocess.Popen(['C:\Program Files\VCG\MeshLab\meshlab.exe', sPath])
    ##TODO error handling
    # try:
    #     ret_code = subprocess.check_call(['ls', '-w'], stdout=subprocess.PIPE, 
    #     stderr=subprocess.PIPE)
    # except subprocess.CalledProcessError as e:
    #     ret_code = e.returncode
    #     print('An error occurred.  Error code:', ret_code)



def writePLY(sPath, aVertices, aColors):
    aColors = aColors.reshape(-1,3)
    aVertices = np.hstack([aVertices.reshape(-1,3), aColors])

    sPLYHeader = '''ply
        format ascii 1.0
        element vertex %(nVertices)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        '''
    with open(sPath, 'w') as oFile:
        oFile.write(sPLYHeader %dict(nVertices=len(aVertices)))
        np.savetxt(oFile, aVertices, '%f %f %f %d %d %d')
        while True:
            if os.access(sPath, os.R_OK):
                openMeshLab(sPath)
                break
 
############################################################################

def navToWelcome(): 
    """ Navigate to previous widget - Welcome Sceen """
    oWelcomeScreen = WelcomeScreen()
    widget.addWidget(oWelcomeScreen)
    widget.setCurrentIndex(widget.currentIndex()+1)

############################################################################
## GUI Screens ###
class WelcomeScreen(QDialog):
    def __init__(self):
        super(WelcomeScreen, self).__init__()
        loadUi("Rekon - Welcome.ui",self)

        self.oCameraCalibrBtn.clicked.connect(self.navToCameraCalibr)
        self.oStereoCalibrBtn.clicked.connect(self.navToStereoCalibr)
        self.oStereoReconstrBtn.clicked.connect(self.navToStereoReconstr)

    def navToCameraCalibr(self):
        """ Navigate to Single Camera Calibration """
        oCameraCalibr = CameraCalibr()
        widget.addWidget(oCameraCalibr)
        widget.setCurrentIndex(widget.currentIndex()+1)

    def navToStereoCalibr(self):
        """ Navigate to Stereo Camera Calibration """
        oStereoCalibr = StereoCalibr()
        widget.addWidget(oStereoCalibr)
        widget.setCurrentIndex(widget.currentIndex()+1)

    def navToStereoReconstr(self):
        """ Navigate to 3D Stereo Reconstruction """
        oStereoReconstr = StereoReconstr()
        widget.addWidget(oStereoReconstr)
        widget.setCurrentIndex(widget.currentIndex()+1)

############################################################################
class CameraCalibr(QDialog):
    def __init__(self):
        super(CameraCalibr, self).__init__()
        loadUi("Rekon - Camera Calibration.ui",self)

        self.oLeftCamBtn.clicked.connect(lambda: self.displayFolderPath("left"))
        self.oRightCamBtn.clicked.connect(lambda: self.displayFolderPath("right"))
        self.oProcessBtn.setEnabled(False)
        self.oProcessBtn.clicked.connect(self.processImages)
        self.oBackBtn.clicked.connect(navToWelcome)
        
    def displayFolderPath(self, sLabel):
        """ Display folder paths under each upload button. Enable "Process Images" button if the user has chosen both folders. """
        sFolderPath = str(QFileDialog.getExistingDirectory(self, "Select Folder"))
        if (sLabel == "left"):
            self.sLeftLabel.setText(sFolderPath)
            self.sLeftFolderPath = sFolderPath
        elif (sLabel == "right"):
            self.sRightLabel.setText(sFolderPath)
            self.sRightFolderPath = sFolderPath

        if (self.sLeftLabel.text() and self.sRightLabel.text()):
            self.oProcessBtn.setEnabled(True)
        else:
            self.oProcessBtn.setEnabled(False)

    def processImages(self): 
        """ Apply single camera calibration on both cameras (left and right) using user input parameters.
        Save coefficients to yml files, default: leftCamParams.yml and rightCamParams.yml. """

        #Apply single camera calibration on each camera
        self.nSquareSize = self.oSquareSizeBox.value()
        self.nChessboardW = self.oChessboardWBox.value()
        self.nChessboardH = self.oChessboardHBox.value()

        ## Check if config files already exist
        ## Warning about overwriting config files 
        # if (path.exists("leftCamParams.yml") or path.exists("rightCamParams.yml")):
        #     oMessageBox = QMessageBox()
        #     oMessageBox.setWindowTitle("Warning")
        #     oMessageBox.setText("Existing camera configuration files will be overwritten.")
        #     oMessageBox.setIcon(QMessageBox.Warning)
        #     oMessageBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Abort)
        #     oMessageBox.setDefaultButton(QMessageBox.Ok)
        #     oMessageBox.buttonClicked.connect(self.warnAboutOverwriting)
        #     oMessageBox.exec_()
        # else:
        #     self.proceedWithCameraCalibr()
        #     oMessageBox.close()
        self.proceedWithCameraCalibr()

    def warnAboutOverwriting(self, i):
        """ Warning about overwriting camera configuration files. If the user aborts, go to previous screen. """
        if(i.text() == "Abort"):
            navToWelcome()
        else: 
            self.proceedWithCameraCalibr()

    def proceedWithCameraCalibr(self):
        """ Apply single camera calibration on each camera (left and right) using user input parameters.
        Save coefficients to yml files (default: leftCamParams.yml and rightCamParams.yml). Display the root mean square (RMS) re-projection error for each camera."""
        ##TODO loading screens 


        retValueL = self.singleCalibration(self.sLeftFolderPath, self.nSquareSize, self.nChessboardW-1, self.nChessboardH-1)
        if (len(retValueL) > 1 ):
            nRMS, K, D, aRotation, aTranslation  = retValueL
            aLeftPath = QFileDialog.getSaveFileName(self, 'Save File', "leftCamParams.yml", "YML Files (*.yml)")
            #Write params to file
            if (aLeftPath[0]):
                saveCameraCoef(K, D, nRMS, aLeftPath[0])
            #Root mean square (RMS) re-projection error, usually it should be between 0.1 and 1.0 pixels in a good calibration.
            print("RMS on left camera calibration: ", nRMS)
        else: 
            ##Error message
            oMessageBox = QMessageBox()
            oMessageBox.setWindowTitle("Error")
            oMessageBox.setText("An error ocurred during left camera calibration. Please try again and make sure that the chessboard pattern is visible in at least 15 photos.")
            oMessageBox.setIcon(QMessageBox.Critical)
            oMessageBox.setStandardButtons(QMessageBox.Ok)
            oMessageBox.setDefaultButton(QMessageBox.Ok)
            oMessageBox.exec_()
            return            


        retValueR = self.singleCalibration(self.sRightFolderPath, self.nSquareSize, self.nChessboardW-1, self.nChessboardH-1)
        if (len(retValueR) > 1 ):
            bSuccessR, nRMS, K, D, aRotation, aTranslation = retValueR
            aRightPath = QFileDialog.getSaveFileName(self, 'Save File', "rightCamParams.yml", "YML Files (*.yml)")
            #Write params to file
            if (aRightPath[0]):
                saveCameraCoef(K, D, nRMS, aRightPath[0])
            #Root mean square (RMS) re-projection error, usually it should be between 0.1 and 1.0 pixels in a good calibration.
            print("RMS on right camera calibration: ", nRMS)
        else:
            ##Error message
            oMessageBox = QMessageBox()
            oMessageBox.setWindowTitle("Error")
            oMessageBox.setText("An error ocurred during right camera calibration. Please try again and make sure that the chessboard pattern is visible in at least 15 photos.")
            oMessageBox.setIcon(QMessageBox.Critical)
            oMessageBox.setStandardButtons(QMessageBox.Ok)
            oMessageBox.setDefaultButton(QMessageBox.Ok)
            oMessageBox.exec_()
            return      


        ##Success message and navigate back home
        if (aLeftPath[0] and aRightPath[0]):
            oMessageBox = QMessageBox()
            oMessageBox.setWindowTitle("Calibration Complete")
            oMessageBox.setText("Camera configuration files have been successfully written.")
            oMessageBox.setIcon(QMessageBox.Information)
            oMessageBox.setStandardButtons(QMessageBox.Ok)
            oMessageBox.setDefaultButton(QMessageBox.Ok)
            oMessageBox.buttonClicked.connect(navToWelcome)
            oMessageBox.exec_()



    def singleCalibration(self, sFolderPath, nSquareSize=0.025, nChessboardW=8, nChessboardH=5):
        """ Single camera calibration using chessboard pattern. Compute RMS, camera matrix, distortion coefficients, rotation and translation vectors. """
        # Array of object 3D points - intersection of squares in the chessboard
        # (0,0,0), (1,0,0), ... etc
        aObjectPoints = np.zeros((nChessboardH*nChessboardW, 3), np.float32)
        aObjectPoints[:, :2] = np.mgrid[0:nChessboardW, 0:nChessboardH].T.reshape(-1, 2)

        aObjectPoints = aObjectPoints * nSquareSize  # Real world coordinates using the nChessboard & nChessboardH of one square

        aSpacePoints = []  # 3D points
        aImagePoints = []  # 2D points

        aPaths = os.listdir(sFolderPath)
        nImages = 0

        try:
            for sImageName in tqdm(aPaths):
                oImg = cv2.imread(os.path.join(sFolderPath, sImageName))
                oBGImg = cv2.cvtColor(oImg, cv2.COLOR_BGR2GRAY)
                
                # cv2.CALIB_USE_INTRINSIC_GUESS
                bFound, aCorners = cv2.findChessboardCorners(oBGImg, (nChessboardW, nChessboardH), flags=cv2.CALIB_CB_ADAPTIVE_THRESH)

                # If found, add object points, image points (after refining them)
                if bFound:
                    aSpacePoints.append(aObjectPoints)

                    aCornersAcc = cv2.cornerSubPix(oBGImg, aCorners, (11, 11), (-1, -1), terminationCriteria)
                    aImagePoints.append(aCornersAcc)

                    print(f"Chessboard found in {sImageName}!")
                    nImages+=1

                    # Draw chessboard corners
                    oImg = cv2.drawChessboardCorners(oImg, (nChessboardW, nChessboardH), aCornersAcc, bFound)
                    cv2.imwrite("draw/" + sImageName, oImg)
                    # cv2.imshow(sImageName, oImg)
                    # cv2.waitKey()
                else:
                    print(f"Chessboard couldn't be detected in  {sImageName}!")

            if (nImages > 15):
                nRMS, aCameraMatrix, aDistorsionCoef, aRotation, aTranslation = cv2.calibrateCamera(aSpacePoints, aImagePoints, oBGImg.shape[::-1], None, None)
                return [True, nRMS, aCameraMatrix, aDistorsionCoef, aRotation, aTranslation]
            else:
                return [False]
        except:
            print("An error occured in single camera calibration")
            return [False]
                

############################################################################

class StereoCalibr(QDialog):
    def __init__(self):
        super(StereoCalibr, self).__init__()
        loadUi("Rekon - Stereo Calibration.ui",self)
        
        self.oLeftCamBtn.clicked.connect(lambda: self.displayFolderPath("left"))
        self.oRightCamBtn.clicked.connect(lambda: self.displayFolderPath("right"))

        self.oLeftCamBtn_2.clicked.connect(lambda: self.displayFolderPath("left-custom"))
        self.oRightCamBtn_2.clicked.connect(lambda: self.displayFolderPath("right-custom"))

        self.oCustomCalibrFilesCb.stateChanged.connect(self.onCustomCalibrFilesCbChecked)

        # Initially hide section about calibration files
        self.sCalibrFilesLabel.hide()
        self.oLeftCamBtn_2.hide()
        self.oRightCamBtn_2.hide()
        self.oLeftCamBtn_2.setEnabled(False)
        self.oRightCamBtn_2.setEnabled(False)

        self.oProcessBtn.setEnabled(False)
        self.oProcessBtn.clicked.connect(self.processImages)
        self.oBackBtn.clicked.connect(navToWelcome)

    def onCustomCalibrFilesCbChecked(self):
        """ Display section about custom calibration files if checkbox is checked. Otherwise, hide section."""
        if self.oCustomCalibrFilesCb.isChecked():
            self.sCalibrFilesLabel.show()

            self.oLeftCamBtn_2.show()
            self.oRightCamBtn_2.show()

            self.oLeftCamBtn_2.setEnabled(True)
            self.oRightCamBtn_2.setEnabled(True)

            self.sLeftLabel_2.show()
            self.sLeftLabel_2.setText("")

            self.sRightLabel_2.show()
            self.sRightLabel_2.setText("")

            self.oProcessBtn.setEnabled(False)
        else:
            self.sCalibrFilesLabel.hide()

            self.oLeftCamBtn_2.hide()
            self.oRightCamBtn_2.hide()

            self.oLeftCamBtn_2.setEnabled(False)
            self.oRightCamBtn_2.setEnabled(False)

            self.sLeftLabel_2.hide()
            self.sRightLabel_2.hide()

            if (self.sLeftLabel.text() and self.sRightLabel.text()):
                self.oProcessBtn.setEnabled(True)
            else: 
                self.oProcessBtn.setEnabled(False)

    def displayFolderPath(self, sLabel): 
        """ Display folder paths/file paths under each upload button. Enable "Process Images" button if the user has chosen both folders. """
       
        if sLabel == "left":
            sFolderPath = str(QFileDialog.getExistingDirectory(self, "Select Folder"))
            self.sLeftLabel.setText(sFolderPath)
            self.sLeftFolderPath = sFolderPath
        elif sLabel == "right":
            sFolderPath = str(QFileDialog.getExistingDirectory(self, "Select Folder"))
            self.sRightLabel.setText(sFolderPath)
            self.sRightFolderPath = sFolderPath
        elif sLabel == "left-custom":
            aFilePath = QFileDialog.getOpenFileName(self, "Select file", "~", "YML Files (*.yml)")
            sFolderPath = aFilePath[0]
            self.sLeftLabel_2.setText(sFolderPath)
            self.sLeftFilePath = sFolderPath
        elif sLabel == "right-custom":
            aFilePath = QFileDialog.getOpenFileName(self, "Select file", "~", "YML Files (*.yml)")
            sFolderPath = aFilePath[0]
            self.sRightLabel_2.setText(sFolderPath)
            self.sRightFilePath = sFolderPath

        if (self.oCustomCalibrFilesCb.isChecked()):
            if (self.sLeftLabel.text() and self.sRightLabel.text() and self.sLeftLabel_2.text() and self.sRightLabel_2.text()):
                self.oProcessBtn.setEnabled(True)
            else:
                self.oProcessBtn.setEnabled(False)
        else: 
            if (self.sLeftLabel.text() and self.sRightLabel.text()):
                self.oProcessBtn.setEnabled(True)
            else:
                self.oProcessBtn.setEnabled(False)

    def processImages(self): 
        """ Apply stereo camera calibration on both cameras (left and right) using user input parameters.
        Save coefficients to stereoCamParams.yml. 
        If configuration file already exists, warn the user about overwritting the file. """

        #Apply stereo camera calibration
        self.nSquareSize = self.oSquareSizeBox.value()
        self.nChessboardW = self.oChessboardWBox.value()
        self.nChessboardH = self.oChessboardHBox.value()

        if (not self.oCustomCalibrFilesCb.isChecked()):
            if (path.exists("leftCamParams.yml") and path.exists("rightCamParams.yml")):
                self.stereoCalibration("leftCamParams.yml", "rightCamParams.yml", self.sLeftFolderPath, self.sRightFolderPath, self.nSquareSize, self.nChessboardW-1, self.nChessboardH-1)
            else: 
                oMessageBox = QMessageBox()
                oMessageBox.setWindowTitle("Error")
                oMessageBox.setText("Default calibration files not found. Please apply single camera calibration or upload custom calibration files.")
                oMessageBox.setIcon(QMessageBox.Critical)
                oMessageBox.buttonClicked.connect(navToWelcome)
                oMessageBox.exec_()
                return
        else:
            self.stereoCalibration(self.sLeftFilePath, self.sRightFilePath, self.sLeftFolderPath, self.sRightFolderPath, self.nSquareSize, self.nChessboardW-1, self.nChessboardH-1)

    def stereoCalibration(self, sLeftFile, sRightFile, sLeftFolderPath, sRightFolderPath, nSquareSize=0.025, nChessboardW=9, nChessboardH=6):
        """ Stereo camera calibration using chessboard pattern. """
        # Array of object 3D points - intersection of squares in the chessboard
        # (0,0,0), (1,0,0), ... etc
        aObjectPoints = np.zeros((nChessboardH * nChessboardW, 3), np.float32)
        aObjectPoints[:, :2] = np.mgrid[0 : nChessboardW, 0 : nChessboardH].T.reshape(-1, 2)

        aObjectPoints = aObjectPoints * nSquareSize  # Real world coordinates using the nChessboard & nChessboardH of one square

        aSpacePoints = []  # 3D points in real world space
        aLeftPoints = []  # 2D points in left image plane.
        aRightPoints = []  # 2D points in right image plane.


        # Get images from folders
        aLeftImgs = os.listdir(sLeftFolderPath);
        aRightImgs = os.listdir(sRightFolderPath);


        if len(aLeftImgs) != len(aRightImgs):
            print("The number of left images doesn't match the number of right images. Images can't be paired.")
            print("Left images count: ", len(aLeftImgs))
            print("Right images count: ", len(aRightImgs))
            # Error pop-up
            oMessageBox = QMessageBox()
            oMessageBox.setWindowTitle("Error")
            oMessageBox.setText("The number of left files doesn't match the number of right files. Images can't be paired.")
            oMessageBox.setIcon(QMessageBox.Critical)
            oMessageBox.buttonClicked.connect(navToWelcome)
            oMessageBox.exec_()
            return

        # Pair the images for single loop handling
        aPairedImages = zip(aLeftImgs, aRightImgs)  

        # Iterate through the pairs and find chessboard corners. Add points to corresponding arrays
        # If openCV can't find the corners, discard the pair.
        nImages = 0
        try:
            for sLeftImg, sRightImg in aPairedImages:
                # Find chessboard corners in each image

                # Left
                oLeftImg = cv2.imread(os.path.join(sLeftFolderPath, sLeftImg))
                oBWLeftImg = cv2.cvtColor(oLeftImg, cv2.COLOR_BGR2GRAY)

                # cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS
                # cv2.CALIB_USE_INTRINSIC_GUESS
                bFoundL, aCornersL = cv2.findChessboardCorners(oBWLeftImg, (nChessboardW, nChessboardH), flags=cv2.CALIB_CB_ADAPTIVE_THRESH)

                # Right 
                oRightImg = cv2.imread(os.path.join(sRightFolderPath, sRightImg))
                oBWRightImg = cv2.cvtColor(oRightImg, cv2.COLOR_BGR2GRAY)

                ##TODO check shape

                # cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS
                # cv2.CALIB_USE_INTRINSIC_GUESS
                bFoundR, aCornersR = cv2.findChessboardCorners(oBWRightImg, (nChessboardW, nChessboardH), flags=cv2.CALIB_CB_ADAPTIVE_THRESH)

                if bFoundL and bFoundR: # Chessboard found in both images
                    nImages +=1
                    # 3D points
                    aSpacePoints.append(aObjectPoints)
                    
                    # Right 2D points
                    aCornersRAcc = cv2.cornerSubPix(oBWRightImg, aCornersR, (11, 11), (-1, -1), terminationCriteria)
                    aRightPoints.append(aCornersRAcc)

                    oImg = cv2.drawChessboardCorners(oRightImg, (nChessboardW, nChessboardH), aCornersRAcc, bFoundR)
                    # cv2.imshow(sRightImg, oImg)
                    # cv2.waitKey()
                    # cv2.imwrite("draw stereo right/" + sRightImg, oImg)

                    # Left 2D points
                    aCornersLAcc = cv2.cornerSubPix(oBWLeftImg, aCornersL, (11, 11), (-1, -1), terminationCriteria)
                    aLeftPoints.append(aCornersLAcc)

                    oImg = cv2.drawChessboardCorners(oLeftImg, (nChessboardW, nChessboardH), aCornersLAcc, bFoundL)
                    # cv2.imshow(sLeftImg, oImg)
                    # cv2.waitKey()
                    # cv2.imwrite("draw stereo left/" + sLeftImg, oImg)
                    print("Chessboard found in image pair: ", sLeftImg, " and ", sRightImg)

                else:
                    print("Chessboard couldn't be detected in image pair: ", sLeftImg, " and ", sRightImg)

            h,w = oBWRightImg.shape 
            K1, D1 = loadCameraCoef(sLeftFile)
            K2, D2 = loadCameraCoef(sRightFile)

            if (nImages > 15):
                nRMS, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(aSpacePoints, aLeftPoints, aRightPoints, K1, D1, K2, D2, (w,h), flags=cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_SAME_FOCAL_LENGTH)
                print("Stereo calibration RMS: ", nRMS)
                R1, R2, P1, P2, Q, roiLeft, roiRigth = cv2.stereoRectify(K1, D1, K2, D2, (w,h), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

                ##Show recfified images to make sure everything's ok
                # sLeftImg = aLeftImgs[0]
                # sRightImg = aRightImgs[0]

                # oLeftImg = cv2.imread(os.path.join(sLeftFolderPath, sLeftImg))
                # cv2.imshow("Left ", oLeftImg)
                # cv2.waitKey()

                # oRightImg = cv2.imread(os.path.join(sRightFolderPath, sRightImg))
                # cv2.imshow("Right", oRightImg)
                # cv2.waitKey()


                # aLeftMapX, aLeftMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w,h), cv2.CV_32FC1)
                # oLeftRectified = cv2.remap(oLeftImg, aLeftMapX, aLeftMapY, cv2.INTER_LINEAR)
                # cv2.imshow("Left rectified", oLeftRectified)
                # cv2.imwrite("rectified/" + sLeftImg, oLeftRectified)
                # cv2.waitKey()
                        

                # aRightMapX, aRightMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w,h), cv2.CV_32FC1)
                # oRightRectified = cv2.remap(oRightImg, aRightMapX, aRightMapY, cv2.INTER_LINEAR)
                # cv2.imshow("Right rectified", oRightRectified)
                # cv2.imwrite("rectified/" + sRightImg, oRightRectified)
                # cv2.waitKey()

                aPath = QFileDialog.getSaveFileName(self, 'Save File', "stereoCamParams.yml", "YML Files (*.yml)")
                if (aPath[0]):
                    saveStereoCoef(K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q, aPath[0])

                    ##Success message and navigate back home
                    oMessageBox = QMessageBox()
                    oMessageBox.setWindowTitle("Stereo Calibration Complete")
                    oMessageBox.setText("Camera configuration files have been successfully written.")
                    oMessageBox.setIcon(QMessageBox.Information)
                    oMessageBox.setStandardButtons(QMessageBox.Ok)
                    oMessageBox.setDefaultButton(QMessageBox.Ok)
                    oMessageBox.buttonClicked.connect(navToWelcome)
                    oMessageBox.exec_()

            else:
                raise ValueError('Not enough image pairs')
        except:
            ##Error message
            oMessageBox = QMessageBox()
            oMessageBox.setWindowTitle("Error")
            oMessageBox.setText("An error ocurred during stereo camera calibration. Please try again and make sure that the chessboard pattern is visible in at least 15 image pairs.")
            oMessageBox.setIcon(QMessageBox.Critical)
            oMessageBox.setStandardButtons(QMessageBox.Ok)
            oMessageBox.setDefaultButton(QMessageBox.Ok)
            oMessageBox.exec_()
            return  

                

############################################################################

class StereoReconstr(QDialog):
    def __init__(self): 
        super(StereoReconstr, self).__init__()
        loadUi("Rekon - Stereo Reconstruction.ui",self)

        # self.oOpenDepthMap.stateChanged.connect(self)
        self.oCustomCalibrFilesCb.stateChanged.connect(self.onCustomCalibrFilesCbChecked)
        self.oCustomQCb.stateChanged.connect(self.onCustomQCbChecked)

        # Initially hide section about calibration file
        self.oUploadBtn.clicked.connect(self.setFilePathLabel)
        self.oUploadBtn.setEnabled(False)
        self.oUploadBtn.hide()
        self.sFilePath.hide()


        #Initailly hide section about custom Q
        self.oCustomQBtn.clicked.connect(lambda: self.setFilePathLabel(True))
        self.oCustomQBtn.setEnabled(False)
        self.oCustomQBtn.hide()
        self.sQFilePath.hide()

        self.oNextBtn.clicked.connect(self.navToNextPage)
        self.oBackBtn.clicked.connect(navToWelcome)

    def onCustomCalibrFilesCbChecked(self):
        """ Display/hide upload & label for stereo calibration file """

        if (self.oCustomCalibrFilesCb.isChecked()):
            self.oUploadBtn.setEnabled(True)
            self.oUploadBtn.show()
            self.sFilePath.show()
            self.sFilePath.setText("")

            self.oNextBtn.setEnabled(False)

        else:
            self.oUploadBtn.setEnabled(False)
            self.oUploadBtn.hide()
            self.sFilePath.hide()

            ##Check if Next button can be enabled based on Q checkbox 
            if (self.oCustomQCb.isChecked()):
                if (self.sQFilePath.text()):
                    self.oNextBtn.setEnabled(True)
                else:
                    self.oNextBtn.setEnabled(False)
            else:
                self.oNextBtn.setEnabled(True)


    def onCustomQCbChecked(self):
        """ Display/hide upload & label for Q  file """

        if (self.oCustomQCb.isChecked()):
            self.oCustomQBtn.setEnabled(True)
            self.oCustomQBtn.show()
            self.sQFilePath.show()
            self.sQFilePath.setText("")

            self.oNextBtn.setEnabled(False)
        else:
            self.oCustomQBtn.setEnabled(False)
            self.oCustomQBtn.hide()
            self.sQFilePath.hide()

            ##Check if Next button can be enabled based on custom stereo file checkbox 
            if (self.oCustomCalibrFilesCb.isChecked()):
                if (self.sFilePath.text()):
                    self.oNextBtn.setEnabled(True)
                else:
                    self.oNextBtn.setEnabled(False)
            else:
                self.oNextBtn.setEnabled(True)


    def setFilePathLabel(self, bQ = False):
        """ Set file path under upload button """

        aFilePath = QFileDialog.getOpenFileName(self, "Select file", "", "YML Files (*.yml)")
        sFilePath = aFilePath[0]
        if (not bQ):

            self.sFilePath.setText(sFilePath)
            self.sFilePathText = sFilePath
            if (sFilePath):
                ##Check if Next button can be enabled based on Q checkbox 
                if (self.oCustomQCb.isChecked()):
                    if (self.sQFilePath.text()):
                        self.oNextBtn.setEnabled(True)
                    else:
                        self.oNextBtn.setEnabled(False)
                else:
                    self.oNextBtn.setEnabled(True)
            else:
                self.oNextBtn.setEnabled(False)

        else: ##triggered by Q
            self.sQFilePath.setText(sFilePath)
            self.sQFilePathText = sFilePath

            if (sFilePath):
                ##Check if Next button can be enabled based on custom stereo file checkbox 
                if (self.oCustomCalibrFilesCb.isChecked()):
                    if (self.sFilePath.text()):
                        self.oNextBtn.setEnabled(True)
                    else:
                        self.oNextBtn.setEnabled(False)
                else:
                    self.oNextBtn.setEnabled(True)
            else:
                self.oNextBtn.setEnabled(False)


    def navToNextPage(self):
        """ Navigate to Parameters screen based on selected algoritghm """
        sAlgorithm = self.oReconstrAlgCb.currentText()
        bOpenDepthMap = self.oOpenDepthMapCb.isChecked()

        if (self.oCustomQCb.isChecked()):
            sQFilePath = self.sQFilePathText
        else: 
            sQFilePath = ''

        if (self.oCustomCalibrFilesCb.isChecked()):
            sFilePath = self.sFilePathText
        else:
            if (path.exists("stereoCamParams.yml")):
                sFilePath = "stereoCamParams.yml"
            else:
                oMessageBox = QMessageBox()
                oMessageBox.setWindowTitle("Error")
                oMessageBox.setText("Default configuration file stereoCamParams.yml not found. Please apply stereo calibration or upload a custom calibration file.")
                oMessageBox.setIcon(QMessageBox.Critical)
                oMessageBox.buttonClicked.connect(navToWelcome)
                oMessageBox.exec_()
                return

        if ("SGBM" in sAlgorithm):
            oSGBMParams = SGBMParams(bOpenDepthMap, sFilePath, sQFilePath)
            widget.addWidget(oSGBMParams)
            widget.setCurrentIndex(widget.currentIndex()+1)
        else:
            oSADParams = SADParams(bOpenDepthMap, sFilePath, sQFilePath)
            widget.addWidget(oSADParams)
            widget.setCurrentIndex(widget.currentIndex()+1)


############################################################################

class SGBMParams(QDialog):
    def __init__(self, bOpenDepthMap, sFilePath, sQFilePath): 
        super(SGBMParams, self).__init__()
        loadUi("Rekon - SGBM Parameters.ui",self)
        self.bOpenDepthMap = bOpenDepthMap
        self.sFilePath = sFilePath
        self.sQFilePath = sQFilePath

        self.oRestoreBtn.clicked.connect(self.restoreDefaultValues)
        self.oRestoreBtn.setIcon(QtGui.QIcon("undo.png"))

        self.oGenerateBtn.setEnabled(False)
        self.oGenerateBtn.clicked.connect(self.proceedWithReconstruction)

        self.oLeftImgBtn.clicked.connect(lambda: self.uploadImage("left"))
        self.oRightImgBtn.clicked.connect(lambda: self.uploadImage("right"))

        self.oBackBtn.clicked.connect(self.navToStereoReconstr)

    def uploadImage(self, sLabel):
        """ Set images (left and right) to corresponding labels """
        if (sLabel == "left"):
            aFilePath = QFileDialog.getOpenFileName(self, "Select image", "", "Image Files (*.png *.jpg *.jpeg)")
            self.sLeftPath = aFilePath[0]
            self.sLeftImage.setStyleSheet(f"background-image : url('{self.sLeftPath}');")
        elif (sLabel == "right"):
            aFilePath = QFileDialog.getOpenFileName(self, "Select image", "", "Image Files (*.png *.jpg *.jpeg)")
            self.sRightPath = aFilePath[0]
            self.sRightImage.setStyleSheet(f"background-image : url('{self.sRightPath }');")

        if (hasattr(self, 'sLeftPath') and hasattr(self, 'sRightPath')):
            if (self.sLeftPath and self.sRightPath):
                self.oGenerateBtn.setEnabled(True)
            else:
                self.oGenerateBtn.setEnabled(False)

    def restoreDefaultValues(self):
        """ Reset default parameter values for SGBM matcher """
        self.oMinDisparity.setValue(-1)
        self.oNumDisparities.setValue(80)
        self.oBlockSize.setValue(3)
        self.oSpeckleWindowSize.setValue(150)
        self.oUniqRatio.setValue(10)
        self.oDisp12MaxDiff.setValue(12)
        self.oPreFilterCap.setValue(63)
        self.oSpeckleRange.setValue(2)

    def proceedWithReconstruction(self):

        nMinDisparity = self.oMinDisparity.value()
        nNumDisparities =  self.oNumDisparities.value()
        nBlockSize = self.oBlockSize.value()
        nSpeckleWindowSize = self.oSpeckleWindowSize.value()
        nUniquenessRatio = self.oUniqRatio.value()
        nDisp12MaxDiff = self.oDisp12MaxDiff.value()
        nPreFilterCap = self.oPreFilterCap.value()
        nSpeckleRange = self.oSpeckleRange.value()


        aDisparity, Q = self.computeDepthMap(self.sFilePath, self.bOpenDepthMap, self.sLeftPath, self.sRightPath, nBlockSize, nMinDisparity, nNumDisparities, nDisp12MaxDiff, nUniquenessRatio, nSpeckleWindowSize, nSpeckleRange, nPreFilterCap)
        
        aColors = cv2.imread(self.sLeftPath, cv2.COLOR_RGB2BGR)
        aColors = cv2.cvtColor(aColors, cv2.COLOR_RGB2BGR)
 
        aMask = aDisparity > aDisparity.min()

        if (self.sQFilePath):
            Q = loadQ(self.sQFilePath)
        print(Q)

        aReprojectedPoints = cv2.reprojectImageTo3D(aDisparity, Q)  

        aFinalPoints = aReprojectedPoints[aMask]
        aFinalColors = aColors[aMask]

        writePLY("reconstructed.ply", aFinalPoints, aFinalColors)

    def navToStereoReconstr(self):
        """ Navigate to 3D Stereo Reconstruction """
        oStereoReconstr = StereoReconstr()
        widget.addWidget(oStereoReconstr)
        widget.setCurrentIndex(widget.currentIndex()+1)

    def computeDepthMap(self, sStereoParams, bShowDepthMap, sLeftImg, sRightImg, nWindowSize=3,nMinDisparity=-1, nNumDisparities=80, nDisp12MaxDiff=12, nUniquenessRatio=10, nSpeckleWindowSize=150, nSpeckleRange=2, nPreFilterCap=63, sMode=cv2.STEREO_SGBM_MODE_SGBM_3WAY):
        """ Compute depth map from image pair and stereo calibration coefficients. """
        try:
            K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q = loadStereoCoef(sStereoParams)  # Get cams params

            oLeftImg = cv2.imread(sLeftImg)
            oRightImg = cv2.imread(sRightImg)

            if (oLeftImg.shape != oRightImg.shape):
                print("Images don't have the same size")
                # Error pop-up
                oMessageBox = QMessageBox()
                oMessageBox.setWindowTitle("Error")
                oMessageBox.setText("Could not compute depth map. Images don't have the same size.")
                oMessageBox.setIcon(QMessageBox.Critical)
                oMessageBox.buttonClicked.connect(navToWelcome)
                oMessageBox.exec_()
                return

            nHeight, nWidth, oChannel = oLeftImg.shape

            # Undistortion and rectification
            ##TODO
            # aLeftMapX, aLeftMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (nWidth, nHeight), cv2.CV_32FC1)
            # oLeftRectified = cv2.remap(oLeftImg, aLeftMapX, aLeftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

            # aRightMapX, aRightMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (nWidth, nHeight), cv2.CV_32FC1)
            # oRightRectified = cv2.remap(oRightImg, aRightMapX, aRightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

            oLeftRectified = oLeftImg
            oRightRectified = oRightImg

            # cv2.imshow("Left rectified", oLeftRectified)
            # cv2.imwrite( sLeftImg + "rectified.jpg" , oLeftRectified)
            # cv2.waitKey()
                    
            # cv2.imshow("Right rectified", oRightRectified)
            # cv2.imwrite(sRightImg + "rectified.jpg", oRightRectified)
            # cv2.waitKey()

            # Convert images to grayscale
            oBWLeft = cv2.cvtColor(oLeftRectified, cv2.COLOR_BGR2GRAY)
            oBWRight = cv2.cvtColor(oRightRectified, cv2.COLOR_BGR2GRAY)

            # SGBM parameters 

            oLeftMatcher = cv2.StereoSGBM_create(
                minDisparity=nMinDisparity,
                numDisparities=nNumDisparities,  # max_disp has to be dividable by 16 f. E. HH 192, 256
                blockSize=nWindowSize,
                P1=8 * 2 * nWindowSize**2,
                P2=32 * 2 * nWindowSize**2,
                disp12MaxDiff=nDisp12MaxDiff,
                uniquenessRatio=nUniquenessRatio,
                speckleWindowSize=nSpeckleWindowSize,
                speckleRange=nSpeckleRange,
                preFilterCap=nPreFilterCap,
                mode=sMode
            )
            
            disparity_map = oLeftMatcher.compute(oBWLeft, oBWRight)

            oFilteredImg = cv2.normalize(src=disparity_map, dst=disparity_map, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
            oFilteredImg = np.uint8(oFilteredImg)

            if (bShowDepthMap):
                cv2.imshow('Disparity/Depth Map', oFilteredImg)
                cv2.waitKey()

            return [disparity_map, Q]
        except:
            # Error pop-up
            oMessageBox = QMessageBox()
            oMessageBox.setWindowTitle("Error")
            oMessageBox.setText("An error occurred while computing the depth map. Please check input data and try again.")
            oMessageBox.setIcon(QMessageBox.Critical)
            oMessageBox.buttonClicked.connect(navToWelcome)
            oMessageBox.exec_()
            return
    

############################################################################
class SADParams(QDialog):
    def __init__(self, bOpenDepthMap, sFilePath, sQFilePath): 
        super(SADParams, self).__init__()
        loadUi("Rekon - SAD Parameters.ui",self)
        self.bOpenDepthMap = bOpenDepthMap
        self.sFilePath = sFilePath
        self.sQFilePath = sQFilePath

        self.oRestoreBtn.clicked.connect(self.restoreDefaultValues)
        self.oRestoreBtn.setIcon(QtGui.QIcon("undo.png"))

        self.oGenerateBtn.setEnabled(False)
        self.oGenerateBtn.clicked.connect(self.proceedWithReconstruction)

        self.oLeftImgBtn.clicked.connect(lambda: self.uploadImage("left"))
        self.oRightImgBtn.clicked.connect(lambda: self.uploadImage("right"))

        self.oBackBtn.clicked.connect(self.navToStereoReconstr)

    def uploadImage(self, sLabel):
        """ Set images (left and right) to corresponding labels """
        if (sLabel == "left"):
            aFilePath = QFileDialog.getOpenFileName(self, "Select image", "~", "Image Files (*.png *.jpg)")
            self.sLeftPath = aFilePath[0]
            self.sLeftImage.setStyleSheet(f"background-image : url('{self.sLeftPath}');")
        elif (sLabel == "right"):
            aFilePath = QFileDialog.getOpenFileName(self, "Select image", "~", "Image Files (*.png *.jpg)")
            self.sRightPath = aFilePath[0]
            self.sRightImage.setStyleSheet(f"background-image : url('{self.sRightPath }');")

        if (hasattr(self, 'sLeftPath') and hasattr(self, 'sRightPath')):
            if (self.sLeftPath and self.sRightPath):
                self.oGenerateBtn.setEnabled(True)
            else:
                self.oGenerateBtn.setEnabled(False)


    def restoreDefaultValues(self): 
        """ Reset default values for SAD algorithm """
        self.oBlockSize.setValue(5)
        self.oSearchBlockSize.setValue(56)

    def navToStereoReconstr(self):
        """ Navigate to 3D Stereo Reconstruction """
        oStereoReconstr = StereoReconstr()
        widget.addWidget(oStereoReconstr)
        widget.setCurrentIndex(widget.currentIndex()+1)
    
    def proceedWithReconstruction(self):
        ##TODO error handling
        try:
            K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q = loadStereoCoef(self.sFilePath)  # Get cams params

            oLeftImg = cv2.imread(self.sLeftPath)
            oRightImg = cv2.imread(self.sRightPath)

            nHeight, nWidth, nChannels = oLeftImg.shape

            # Undistortion and rectification
            # aLeftMapX, aLeftMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (nWidth, nHeight), cv2.CV_32FC1)
            # oLeftRectified = cv2.remap(oLeftImg, aLeftMapX, aLeftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

            # aRightMapX, aRightMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (nWidth, nHeight), cv2.CV_32FC1)
            # oRightRectified = cv2.remap(oRightImg, aRightMapX, aRightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

            # Convert images to grayscale
            oLeftRectified = oLeftImg
            oRightRectified = oRightImg 

            # oBWLeft = cv2.cvtColor(oLeftRectified, cv2.COLOR_BGR2GRAY)
            # oBWRight = cv2.cvtColor(oRightRectified, cv2.COLOR_BGR2GRAY)

            npLeft = np.asarray(oLeftRectified)
            npRight = np.asarray(oRightRectified)

            aLeftImg = npLeft.astype(int)
            aRightImg = npRight.astype(int)

            aDisparity = self.computeDepthMapSAD(aLeftImg, aRightImg)

            aColors = cv2.imread(self.sLeftPath, cv2.COLOR_RGB2BGR)
            aColors = cv2.cvtColor(aColors, cv2.COLOR_RGB2BGR)
    
            aMask = aDisparity > aDisparity.min()
            if (self.sQFilePath): 
                Q = loadQ(self.sQFilePath)
            print(Q)
            
            aReprojectedPoints = cv2.reprojectImageTo3D(aDisparity, Q) 
            print(aReprojectedPoints) 

            aFinalPoints = aReprojectedPoints[aMask]
            aFinalColors = aColors[aMask]

            writePLY("reconstructed.ply", aFinalPoints, aFinalColors)
        except:
            # Error pop-up
            oMessageBox = QMessageBox()
            oMessageBox.setWindowTitle("Error")
            oMessageBox.setText("An error occurred while computing the reconstruction. Please check the input data and try again.")
            oMessageBox.setIcon(QMessageBox.Critical)
            oMessageBox.buttonClicked.connect(navToWelcome)
            oMessageBox.exec_()
            return

    def computeSumAbsDiff(self,aLeftBlock, aRightBlock):
        """ Sum of absolute difference between 2 pixel blocks """
        if aLeftBlock.shape != aRightBlock.shape:
            return -1

        return np.sum(abs(aLeftBlock - aRightBlock))

    def compareBlocks(self, nRow, nCol, aLeftBlock, aRightImg, nBlockSize=5, nSearchBlockSize=56):
        """
        Compare left block of pixels with multiple blocks from the right
        image using nSearchBlockSize to limit the search in the right
        image.
        Returns (row, column) row and column index of the best matching block 
        in the right image
        """
        # Get search range for the right image
        nMinCol = max(0, nCol - nSearchBlockSize)
        nMaxCol = min(aRightImg.shape[1], nCol + nSearchBlockSize)

        bFirst = True
        nMinSad = None
        pMinSAD = None

        for nCol in range(nMinCol, nMaxCol):

            aRightBlock = aRightImg[nRow: nRow + nBlockSize, nCol: nCol + nBlockSize]
            nSAD = self.computeSumAbsDiff(aLeftBlock, aRightBlock)

            if bFirst:
                nMinSad = nSAD
                pMinSAD = (nRow, nCol)
                bFirst = False
            else:
                if nSAD < nMinSad:
                    nMinSad = nSAD
                    pMinSAD = (nRow, nCol)

        return pMinSAD

    def computeDepthMapSAD(self, aLeftImg, aRightImg, nBlockSize=5, nSearchBlockSize=56):

        if aLeftImg.shape != aRightImg.shape:
            print("Images don't have the same size")
            # Error pop-up
            oMessageBox = QMessageBox()
            oMessageBox.setWindowTitle("Error")
            oMessageBox.setText("Could not compute depth map. Images don't have the same size.")
            oMessageBox.setIcon(QMessageBox.Critical)
            oMessageBox.buttonClicked.connect(navToWelcome)
            oMessageBox.exec_()
            return


        nHeight, nWidth, nChannels = aLeftImg.shape
        aDisparity = np.zeros((nHeight, nWidth))

        # Go over each pixel position
        for nRow in tqdm(range(nBlockSize, nHeight-nBlockSize), desc = "Computing depth map"):
            for nCol in range(nBlockSize, nWidth-nBlockSize):
                aLeftBlock = aLeftImg[nRow : nRow + nBlockSize, nCol : nCol + nBlockSize]
                pMinSAD = self.compareBlocks(nRow, nCol, aLeftBlock, aRightImg, nBlockSize, nSearchBlockSize)
                aDisparity[nRow, nCol] = abs(pMinSAD[1] - nCol)
        if (self.bOpenDepthMap):
            plt.imshow(aDisparity, cmap='hot', interpolation='nearest')
            # plt.savefig('disparity.png')
            plt.show()
            # img = PIL.Image.fromarray(aDisparity, 'L')
            # img.show() 
        
        return np.uint8(aDisparity)

############################################################################
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon('icon.png'))
    welcome = WelcomeScreen()
    widget = QtWidgets.QStackedWidget()
    widget.addWidget(welcome)
    widget.setFixedWidth(1000);
    widget.setFixedHeight(800);
    widget.show()
    sys.exit(app.exec_())
