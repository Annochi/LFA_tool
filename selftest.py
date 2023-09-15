#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Script name: selftest.py
# Created on: 2022-03-02
# Author: Andreas Postel
# Purpose:
#   Read database(s) (if provided) and run two Proteus models with acquired data
#   the results are compared to check for differences in the calculation.
#   If no database is provided, the CSV files in the data directory are used.
# Usage:
#   selftest.py [-h] [--analyze] [--calc] [--database DATABASE] [-m]
#               [--plot] [--tolerance TOLERANCE] [--calcref] [--clean]
#               [--code {0,1}] [--compall] [--model {0,1,2}]
import threading

# Import stuff
from babel.numbers import format_decimal
from scipy import interpolate
from pathlib import Path
import argparse
import csv
import glob
import hashlib
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sqlite3
import struct
import subprocess
import sys
import shutil
import re
import time
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QLineEdit
from PyQt5.QtCore import pyqtSignal, QObject, pyqtSlot
import multiprocessing



##############
# Parameters #
##############

CMD = "CalcDiffusConsole/CalcDiffusNoGui.exe"       # path to LFA NoGui reference
CMD2 = "CalcDiffusConsole/TT_Tool_Console.exe"      # path to second LFA NoGui
DATAPATH = "data/"                      # directory for storing data from database(s)
FIGPATH = "plots/"                      # directory for plots
OUTPUTPATH = "output/"                  # directory for output data
REFOUTPUT = "reference/"                # subdirectory for the results from LFA reference version
NEWOUTPUT = "new/"                      # subdirectory for the results from LFA test versionca
MONOPATH = "/usr/bin/mono"              # path to mono if running on UNIX/Linux platform
LOGPATH = "log/"                        # directory for logfile
LOGFILE = Path(LOGPATH) / "logfile.log"
DEVRESULS = "results/"
DBPATH = "DBs/"
# Tolerance for warning if difference is significant - default 5e-3 (0.5%)
# You can also overwrite this by an optional argument
TOLERANCE = 5e-3

REFOUTPUT = OUTPUTPATH + REFOUTPUT
NEWOUTPUT = OUTPUTPATH + NEWOUTPUT
# list of all directories that are needed for operation
DIRS = [DATAPATH, FIGPATH, OUTPUTPATH, REFOUTPUT, NEWOUTPUT, LOGPATH, DEVRESULS]

class Worker(QtCore.QObject):
    updateProgress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, fn):
        super(Worker, self).__init__()
        self.fn = fn

    @pyqtSlot()
    def run(self):
        self.fn()
        self.finished.emit()

class Worker1(QtCore.QObject):
    updateProgress = pyqtSignal(int)
    finished = pyqtSignal()

    @pyqtSlot(int)
    def run1(self):
        for i in range(101):
            time.sleep(1)
            self.updateProgress.emit(i)

        self.finished.emit()


class Ui_MainWindow(QtWidgets.QMainWindow):
    """Class for building  GUI """
    Progress = pyqtSignal(int)

    def __init__(self):
        super(Ui_MainWindow, self).__init__()

        self.centralwidget = None
        #self.progressBar = None
        self.calcButton = None
        self.worker = None
        self.worker_thread = None
        self.CMD = "CalcDiffusConsole/CalcDiffusNoGui.exe"
        self.CMD2 = "CalcDiffusConsole/TT_Tool_Console.exe"
        #self.threadpool = QtCore.QThreadPool()
        self.current_value = 0
        self.progressBar = QtWidgets.QProgressBar()
        self.Progress.connect(self.progressBar.setValue)

    def setupUi(self, MainWindow):
        """Create MainWindow appearance"""

        self.MainWindow = MainWindow
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 900)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.checkBox1 = QtWidgets.QCheckBox('model 0', self.centralwidget)
        self.checkBox1.setToolTip('00_StandardModel')
        self.checkBox1.setGeometry(QtCore.QRect(20, 115, 100, 20))

        self.checkBox2 = QtWidgets.QCheckBox('model 1', self.centralwidget)
        self.checkBox2.setToolTip('01_TransparentPModel')
        self.checkBox2.setGeometry(QtCore.QRect(20, 140, 100, 20))

        self.checkBox3 = QtWidgets.QCheckBox('model 2', self.centralwidget)
        self.checkBox3.setToolTip('02_PenetrationModel')
        self.checkBox3.setGeometry(QtCore.QRect(20, 165, 100, 20))

        self.checkBox4 = QtWidgets.QCheckBox('model 3', self.centralwidget)
        self.checkBox4.setToolTip('03_InplaneRadialModel_isotropic')
        self.checkBox4.setGeometry(QtCore.QRect(20, 190, 100, 20))

        self.checkBox5 = QtWidgets.QCheckBox('model 4', self.centralwidget)
        self.checkBox5.setToolTip('04_InplaneRadialModel_anisotropic')
        self.checkBox5.setGeometry(QtCore.QRect(20, 215, 100, 20))

        self.calcButton = QtWidgets.QPushButton(self.centralwidget)
        self.browseButton = QtWidgets.QPushButton(self.centralwidget)
        self.browseButton1 = QtWidgets.QPushButton(self.centralwidget)

        self.calcButton.setText('Start')
        self.calcButton.setGeometry(150, 650, 100, 40)
        self.browseButton.setText('Browse')
        self.browseButton.setGeometry(320, 500, 100, 20)
        self.browseButton1.setText('Browse')
        self.browseButton1.setGeometry(320, 525, 100, 20)

        self.pathtext = QLineEdit(self.centralwidget)
        self.pathtext1 = QLineEdit(self.centralwidget)
        self.pathtext.setGeometry(20, 500, 300, 20)
        self.pathtext1.setGeometry(20, 525, 300, 20)


        # self.text_browser = QtWidgets.QTextBrowser(self.centralwidget)
        # self.text_browser.setGeometry(QtCore.QRect(400, 115, 600, 300))

        self.text_browser1 = QtWidgets.QTextBrowser(self.centralwidget)
        self.text_browser1.setGeometry(QtCore.QRect(500, 580, 680, 300))

        self.table = QtWidgets.QTableWidget(self.centralwidget)
        self.table.setGeometry(QtCore.QRect(500, 115, 680, 450))



        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)

        self.progressBar.setGeometry(30, 50, 340, 30)
        #self.progressBar.setValue(0)

        self.extime = QtWidgets.QLabel(self.centralwidget)
        self.extime.setGeometry(10, 850, 200, 30)

        self.worker = Worker(self.start)
        self.worker1 = Worker1()

        self.thread = QtCore.QThread()
        self.thread1 = QtCore.QThread()




        self.Progress.connect(self.worker.run)
        self.Progress.connect(self.worker1.run1)
        self.worker.finished.connect(self.calculationFinished)

        #self.worker1.updateProgress.connect(self.progressBar.setValue)
        self.worker1.updateProgress.connect(self.onupdateProgress)

        self.worker.moveToThread(self.thread)
        self.worker1.moveToThread(self.thread1)
        self.thread.start()
        self.thread1.start()

        MainWindow.setCentralWidget(self.centralwidget)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def set_up_connection(self):

        self.browseButton.clicked.connect(self.fileopen)
        self.browseButton1.clicked.connect(self.fileopen)
        self.calcButton.clicked.connect(self.startCalculation)



    def onupdateProgress(self):
        if self.current_value <= self.progressBar.maximum():
            self.current_value += 5
            self.progressBar.setValue(self.current_value)



    def calculationFinished(self):
        # Re-enable the "Start" button when the calculation is finished

        self.writefinal()
        self.progressBar.setValue(100)
        self.calcButton.setEnabled(True)


    def startCalculation(self):

        self.calcButton.setEnabled(False)
        n = 50
        self.progressBar.setMaximum(n)
        self.Progress.emit(n)


    # def updateProgressBar(self, maxVal):
    #     self.progressBar.setValue(self.progressBar.value() + maxVal)
    #     if maxVal == 0:
    #         self.progressBar.setValue(100)




    def fileopen(self):
        """Open the reference and new LFA NoGui"""

        filename = QFileDialog.getOpenFileName(None, 'File', filter='*.exe')
        if filename:
            # directory_path = os.path.dirname(str(filename))

            if self.MainWindow.sender() == self.browseButton:
                self.pathtext.setText(str(filename))
                self.CMD = filename[0]
            else:
                self.pathtext1.setText(str(filename))
                self.CMD2 = filename[0]

        return self.CMD, self.CMD2

    def start(self):
        st = time.time()

        # self.worker = Worker()
        # self.worker.updateProgress.connect(self.onupdateProgress)
        # self.worker.start()
        try:
            processes = []
            create_subdirs(DIRS, True, None)
            check_dict = [self.checkBox1, self.checkBox2, self.checkBox3, self.checkBox4, self.checkBox5]
            #self.startCalculation()
            # Define what's model is checked
            for i in range(len(check_dict)):
                checkbox = check_dict[i]
                if checkbox.isChecked():

                    self.performCalculationsForModel(i)

            #self.writefinal()

        except Exception as e:
            error = QtWidgets.QErrorMessage(self.centralwidget)
            error.showMessage(str(e))
        et = time.time()
        elapsed_time = et - st
        self.extime.setText("%.3f" % elapsed_time)

    def performCalculationsForModel(self, m):
        """Read dbs for checked models and do calculation """

        args.model = m
        args.compall = True

        # get reference db files from directory DBs
        try:
            # get a list of all reference models
            subdirectories = [d for d in os.listdir(DBPATH) if os.path.isdir(os.path.join(DBPATH, d))]
            model_directory = [d for d in subdirectories if d.startswith(str(m))]
            for model in model_directory:
                path = os.path.join(DBPATH, model)

                # list of all db paths in model's directory
                db_paths = [os.path.join(path, filename) for filename in os.listdir(path) if
                              os.path.isfile(os.path.join(path, filename))]

                # list of db names
                dbs = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

                # do calculation for each of db within one model
                for i, db_file in enumerate(db_paths):
                    db = sqlite3.connect(db_file)
                    name = dbs[i]
                    logging.info("Reading Database: {0}".format(name))
                    read_database(db, name, None)
                    self.calculate()
                    create_subdirs(DIRS, True, None)

        except Exception as e:
            error = QtWidgets.QErrorMessage(self.centralwidget)
            error.showMessage(str(e))

    def calculate(self):



        logging.info("Calculating diffusivity for reference version")
        # t = threading.Thread(target=calc_diffusivity, args=(self.CMD, DATAPATH, REFOUTPUT))
        # t.start()
        calc_diffusivity(self.CMD, DATAPATH, REFOUTPUT)

        logging.info("Calculating diffusivity for new Proteus version")
        # t1 = threading.Thread(target=calc_diffusivity, args=(self.CMD2, DATAPATH, REFOUTPUT))
        # t1.start()
        calc_diffusivity(self.CMD2, DATAPATH, NEWOUTPUT)

        # t.join()
        # t1.join()
        group_results((NEWOUTPUT, REFOUTPUT))

        logging.info("Comparing results")
        compare_results(REFOUTPUT, NEWOUTPUT, args.compall)

    def writefinal(self):

        """Read data from temporary files , create and display 2 result files"""

        all = pd.DataFrame()
        models = pd.DataFrame()
        model_numbers = []
        for filename in os.listdir(FIGPATH):
            if filename.endswith(".csv"):
                file_path = os.path.join(FIGPATH, filename)

                # Read CSV into a DataFrame
                df = pd.read_csv(file_path)

                # Get the first model number (it's the same for all rows in the file)
                model_number = df.iloc[0]['#Model']
                model_numbers.append(model_number)

                # Separate temporary.csv into all shots result and average dev within model
                all = pd.concat([all, df.iloc[:, :6]], ignore_index=True)
                last_two_columns = df.iloc[:, -2:].dropna()

                last_two_columns.insert(0, "Model", f"Model #{model_number}")
                models = pd.concat([models, last_two_columns], ignore_index=True)

        create_subdirs(DIRS, True, '.csv')

        all.to_csv(DEVRESULS + "all.csv",  index=False)
        models.to_csv(DEVRESULS + "avg models.csv", index=False)

        all_sorted = all.sort_values(by='#Deviation', ascending=False)
        all_sorted.to_csv(DEVRESULS + "sorted.csv", index=False)

        try:
            # Display the CSV content in the GUI
            self.table.setRowCount(len(all))
            self.table.setColumnCount(len(all.columns))
            self.table.setHorizontalHeaderLabels(all.columns)

            for row_index, row_data in all.iterrows():
                for col_index, cell_data in enumerate(row_data):
                    item = QtWidgets.QTableWidgetItem(str(cell_data))
                    self.table.setItem(row_index, col_index, item)
            self.table.setColumnWidth(5, 200)

            #content = all.to_string(index=False)
            content1 = models.to_string(index=False)

            #self.text_browser.setPlainText(content)
            self.text_browser1.setPlainText(content1)
        except Exception as e:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            self.text_browser1.setPlainText("Error reading CSV file: " + str(e))

        logging.info('Process is finished')


class Data:
    """Class for shot data in the database"""

    def __init__(self, dbname, dtime, detector, ptime, pulse, material, temp,
                 diameter, d0, d1, d2, thickness_t, thickness_rt, cp_t,
                 density_t, thermal_diff, datetime, fintemp, spotsize,
                 isMultilayer, layerData=None):
        # TODO split into physical, instrumentational and other parameters
        self.detector = detector
        self.dtime = dtime
        self.pulse = pulse
        self.ptime = ptime
        self.material = material
        self.temp = temp
        self.datetime = datetime
        self.fintemp = fintemp

        # additional parameters that are required by the mathematical model
        self.dbname = dbname
        self.diameter = diameter
        self.d0 = d0
        self.d1 = d1
        self.d2 = d2
        self.thickness_t = thickness_t
        self.thickness_rt = thickness_rt
        self.cp_t = cp_t
        self.density_t = density_t
        self.thermal_diff = thermal_diff
        self.spotsize = spotsize
        self.isMultilayer = isMultilayer
        self.layerData = layerData


class LayerData:
    def __init__(self, idSample, layerIndex, layerName, thickness,
                 materialName, refTemp, refTempDensity, isUnknown,
                 matProperties) -> None:
        self.idSample = idSample
        self.layerIndex = layerIndex
        self.thickness = thickness
        self.materialName = materialName
        self.refTemp = refTemp
        self.refTempDensity = refTempDensity
        self.matProperties = matProperties
        self.isUnknown = isUnknown
        self.layerName = layerName


class Measurement:
    """Class for comparison of LFA diffusivity results"""

    def __init__(self, material, version_1, version_2):
        self.material = material
        self.versions = [version_1, version_2]
        self.diffusivities = [[], []]
        self.std_devs = [[], []]
        self.temperatures = [[], []]
        self.differences = None
        self.maxdifference = None
        self.tempstep = None
        self.dbname = None

    def adddata(self, diffusivities_1, diffusivities_2, std_devs_1, std_devs_2,
                temperatures_1, temperatures_2, tempstep, dbname):
        """Add data to the measurement."""
        for i in range(len(diffusivities_1)):
            self.diffusivities[0].append(diffusivities_1[i])
            self.diffusivities[1].append(diffusivities_2[i])
            self.std_devs[0].append(std_devs_1[i])
            self.std_devs[1].append(std_devs_2[i])
            self.temperatures[0].append(temperatures_1[i])
            self.temperatures[1].append(temperatures_2[i])
            self.tempstep = tempstep
            self.dbname = dbname

    def calcdifferences(self):
        """Calculate differences between diffusivities for all data
        and maximum difference in the sample."""
        # if there is more than one diffusivity (i.e. temperature)
        try:
            self.differences = []
            for i in range(len(self.diffusivities)):
                linedifference = abs(self.diffusivities[1][i] / self.diffusivities[0][i] - 1)
                self.differences.append(linedifference)
            self.maxdifference = max(self.differences)
        # if there is only one diffusivity
        except IndexError:
            self.maxdifference = self.differences[0]

        return self.maxdifference

    def printdifference(self):
        """Calmeasurement.materialculate differences among diffusivities and print maximum difference."""
        dbname = self.dbname
        material = self.material
        tempstep = self.tempstep
        version_1 = self.versions[0]
        version_2 = self.versions[1]
        maxdifference = self.maxdifference
        if self.maxdifference is not np.isnan(self.maxdifference):
            print("Database:             {0}".format(dbname))
            print("Material:             {0}".format(material))
            print("Tempstep:             {0}".format(tempstep))
            print("Proteus version 1:    {0}".format(version_1))
            print("Proteus version 2:    {0}".format(version_2))
            print("Max. difference:      {0:.2e}".format(maxdifference))

        if maxdifference >= TOLERANCE:
            print("Difference significant!")


def writedeviation(diff, measurement, m):
        """Print the deviation in new csv file in the directory /results"""

        try:
            db = measurement[0].dbname
            material = [measurement.material for measurement in measurement]
            temp = [measurement.tempstep for measurement in measurement]


            #check if difference is nan and exclude from the list
            dif = [item for item in diff if not np.isnan(item)]
            avgdeviation = np.mean(dif)
            maxdeviation = np.max(dif)

            #shots = [[index + 1 for index, item in enumerate(diff)]  in measurement]

            temporay_shots = []
            current_name = None
            current_index = 0


            for index, name in enumerate(material):
                if name != current_name:
                    current_name = name
                    current_index = 1
                    temporay_shots.append(str(current_index))
                else:
                    current_index += 1  # Update current index
                    temporay_shots[-1] = temporay_shots[-1] + ' ' + str(current_index)
                    #temporay_shots[-1] = ' '.join([temporay_shots[-1]] + [str(current_index).split()])

            # Construct the 'shots' list and split it into individual values
            split_shots = ' '.join(temporay_shots)

            # Use regular expression to match multi-digit numbers
            matches = re.findall(r'\d+', split_shots)

            # Convert the matches to a list of integers
            shots = [int(match) for match in matches]
            resname = FIGPATH + str(m) +'_temporary.csv'

            data = {'#DB': db, '#Model': m, '#Mment': material, '#Shots': shots, '#Temp': temp, '#Deviation': diff, '#MaxDeviation': maxdeviation, '#AvgDeviation': avgdeviation}

            df = pd.DataFrame(data)

            # Remove duplicates from #AvgDeviation column and write to deviation.csv
            df.loc[df['#AvgDeviation'].duplicated(), '#AvgDeviation'] = np.nan
            df.loc[df['#MaxDeviation'].duplicated(), '#MaxDeviation'] = np.nan
            #df.loc[df['#DB'].duplicated(), '#DB'] = np.nan

            # Check if the file already exists
            file_exists = os.path.exists(resname)

            # Append data to the file without writing headers
            df.to_csv(resname, mode='a', index=False, header=not file_exists)

            if file_exists:
                df = pd.read_csv(resname)
                deviation = df['#Deviation']  # Replace with your actual column name
                dif = [item for item in deviation if not np.isnan(item)]
                avgdeviation = np.mean(dif)
                maxdeviation = np.max(dif)

                # Create a new DataFrame with statistics
                stats_df = pd.DataFrame({'#MaxDeviation': [maxdeviation], '#AvgDeviation': [avgdeviation]})


                # Append the statistics DataFrame to the original DataFrame
                #df.drop(['#MaxDeviation', '#AvgDeviation'])
                del (df['#MaxDeviation'] , df['#AvgDeviation'])
                df = pd.concat([df, stats_df], axis=1)

                # Save the updated DataFrame to the CSV file
                df.to_csv(resname, index=False)




            # df.to_csv(resname, mode='w' if not os.path.exists(resname) else 'a', index=False)

            # for filename in os.listdir(FIGPATH):
            #     if filename.endswith(str(m) +'_temporary.csv') :
            #         df.to_csv(resname, mode='a', index=False, header=False)
            # else:




        except Exception as e:
            logging.info("At least one of folders new/reference empty - comparing failed ")
            logging.info(str(e))



def bspline(blob, point):
    """Calculate B-spline for 1D datablob curve and return interpolated value at given point

    If a B-spline cannot be calculated, because the datablob only contains a
    single value (e.g. the expansion is zero over the entire temperature
    range), the return value will be zero.
    """
    t, value = convert_data(blob, 'd')
    try:
        tck = interpolate.splrep(t, value)
        returnvalue = interpolate.splev(point, tck)
    except TypeError:
        returnvalue = 0
    return returnvalue


def calc_density(rho_tref, alpha_tref, alpha_tcurr):
    """Calculate density at current temperature

    Parameters
    ----------
    rho_ref
        Density at reference temperature
    alpha_tref
        Thermal expansion at reference temperature
    alpha_curr
        Thermal expansion at current temperature

    Notes
    -----
    .. math::
        \\rho (T_{curr}) = \\rho(T_{ref}) \\times
        \\left(\\frac{1 + \\alpha(T_{ref})}{1 + \\alpha(T_{curr})}\\right)^3
    """
    rho_curr = rho_tref * ((1 + alpha_tref) / (1 + alpha_tcurr))**3
    return rho_curr


def calc_diffusivity(path_calculator, path_data, output):
    """Calculate the diffusivity for all csv files in given directory"""
    calcerror = False
    for file in glob.glob(path_data + '**/*detector.csv', recursive=True):
        logging.info("Processing file {0}".format(str(file)))
        detectorfile = file
        pulsefile = Path(detectorfile.replace("detector", "pulse"))

        outputfile = detectorfile.replace(str(Path(path_data)), str(Path(output)))
        detectorfile = Path(detectorfile)
        outputfile = Path(outputfile.replace("detector", "output"))
        outputdir = os.path.dirname(outputfile)
        create_subdirs([outputdir], False, None)
        command = [path_calculator, str(args.code), str(args.model),
                   str(args.pulse), str(args.baseline), str(pulsefile),
                   str(detectorfile), str(outputfile)]

        if args.mono:
            command = [MONOPATH] + command
        ret = subprocess.call(command)

        if ret != 0:
            logging.error("Calculation for {0} failed with return"
                          " code {1}".format(detectorfile, ret))
            calcerror = True

    return calcerror


def calc_thickness(thickness_rt, scl):
    """Calculate real thickness using the thermal expansion and room temperature thickness"""
    return thickness_rt * (1 + scl)


def compare_results(path_reference, path_newoutput, comp_all):
    """Parse path_reference for csv files. If they exist also in path_NEWOUTPUT, compare results

    Parameters
    ----------
    path_reference
        Path to directory with reference results
    path_newoutput
        Path to directory with new results

    Returns
    -------
    None.

    """
    # parse path_reference for .csv files
    # if files exist in both directories, add them to the filelist
    files = []
    calcerror = False
    # only if all files are compared
    if comp_all:
        for file in glob.glob(path_reference + "/" + "*/*/*.csv"):
            filename = str(os.path.relpath(file, path_reference))
            if os.path.isfile(path_newoutput + filename):
                files.append(filename)
            else:
                calcerror = True
                logging.warning("Warning: No matching file for " + path_newoutput + filename)
    else:
        for file in glob.glob(path_reference + "/" + "*/*summary.csv"):
            filename = str(os.path.relpath(file, path_reference))
            if os.path.isfile(path_newoutput + filename):
                files.append(filename)
            else:
                calcerror = True
                logging.warning("Warning: No matching file for " + path_newoutput + filename)

    # parse all csv tuples for relevant information
    # (Material, Diffusivity, Std_Dev, temperature, proteus version)
    measurements = []
    for file in files:
        measurements.append(dataset(path_reference, path_newoutput, file))

    for measure in measurements:
        measure.printdifference()
        print()

    # get num of model and create a list of measurement differences

    diff = []

    if args.model:
        model = args.model
    else:
       model = "standard"
    for measure in measurements:
        diff.append(measure.calcdifferences())
    writedeviation(diff, measurements, model)

    return calcerror


def convert_data(datablob, datatype):
    """Convert blob into data type pair and return it as two arrays (time/temperature and value)"""
    time = []
    value = []
    for counter, item in enumerate(struct.iter_unpack(datatype, datablob)):
        if counter % 2 == 0:
            time.append(item[0])
        else:
            value.append(item[0])
    time = np.array(time)
    value = np.array(value)
    return time, value


def create_subdirs(directories, clean, extension):
    """Create subdirectories if not existing yet and clean them from old files, if requested"""
    for directory in directories:
        if not os.path.isdir(directory):
            try:
                path = Path(directory)
                path.mkdir(parents=True)
            except OSError:
                print("Creation of the directory %s failed" % directory)
        # clean subdirectory from (old) png and csv files
        if clean:
            if extension:
                for f in os.listdir(FIGPATH):
                    if f.endswith(extension):
                        pic = os.path.join(FIGPATH, f)  # Get the full file path
                        os.remove(pic)

            elif DATAPATH and REFOUTPUT and NEWOUTPUT:
                folder_paths = [DATAPATH, REFOUTPUT, NEWOUTPUT]

                for folder_path in folder_paths:
                    if os.path.isdir(folder_path):
                        # List all files and subdirectories in the folder
                        items_to_delete = os.listdir(folder_path)
                        for item in items_to_delete:
                            item_path = os.path.join(folder_path, item)
                            if os.path.isfile(item_path):
                                os.remove(item_path)  # Remove file
                            elif os.path.isdir(item_path):
                                shutil.rmtree(item_path)

            elif DEVRESULS:
                pass


def dataset(dir1, dir2, file):
    """Create dataset composed of two measurements"""
    material, diffusivities_2, std_devs_2, temperatures_2, version_2, tempstep, \
    dbname = read_data(dir2, file)
    material, diffusivities_1, std_devs_1, temperatures_1, version_1, tempstep,\
        dbname = read_data(dir1, file)
    measure = Measurement(material, version_1, version_2)
    measure.adddata(diffusivities_1, diffusivities_2, std_devs_1, std_devs_2, temperatures_1,
                    temperatures_2, tempstep, dbname)
    measure.calcdifferences()
    return measure


def group_results(paths):
    """Summarize results from individual shots by FinalTemperature"""
    for path in paths:
        for measurement in glob.glob(path + '*/'):
            for tempstep in glob.glob(measurement + '*/'):
                files = glob.glob(tempstep + '*.csv')
                # skip empty directories
                if len(files) == 0:
                    break
                dbname = None
                diffs = []
                material = None
                std_devs = []
                temp = None

                # read all files for entire tempstep
                for file in files:
                    result = read_data(None, file)
                    # only read-out material once, since it is always the same
                    if not material:
                        material = result[0]
                    diffs.append(result[1])
                    std_devs.append(result[2])
                    if not temp:
                        temp = result[3][0]
                    if not dbname:
                        dbname = result[5]

                # calculate std_dev for mean diff by gaussian error propagation
                diff = np.mean(diffs)
                std_devs = np.array(std_devs)
                items = len(std_devs)
                std_devs_square = np.square(std_devs)
                std_dev = 1/items * np.sqrt(np.sum(std_devs_square))

                # write results to output file
                lastdir = os.path.basename(os.path.dirname(tempstep))
                outputname = measurement + lastdir + "_summary.csv"
                write_result(material, diff, std_dev, temp, outputname, lastdir, dbname)


def read_data(directory, filepath):
    """Read csv file and return relevant information"""
    # this is run if called by dataset
    try:
        version = directory.removesuffix("/")
        filename = os.path.basename(filepath)

        if args.compall:
            tempstep = os.path.basename(os.path.dirname(filepath))
        else:
            tempstep = filename.split('_')[0]

    # this is run if called by group_results
    except AttributeError:
        version = None
        directory = "."
        tempstep = os.path.basename(os.path.dirname(filepath))
    temperature = []
    diffusivity = []
    std_dev = []
    # actually read the file
    with open(directory + "/" + filepath, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        relevantdata = False
        for row in reader:
            try:
                if row[0] == "#Database":
                    dbase = row[1]
                if row[0] == "#Material":
                    material = row[1]
                # skip reading when encountering fit data
                elif row[0] == "##Response_fit":
                    break
                elif relevantdata is True:
                    temperature.append(float(
                        row[1]))
                    diffusivity.append(float(
                        row[2]))
                    std_dev.append(float(
                        row[3]))
                elif row[0].startswith('#Shot'):
                    relevantdata = True
            except IndexError:
                pass
    if version:
        return material, diffusivity, std_dev, temperature, version, tempstep, dbase
    else:
        return material, diffusivity, std_dev, temperature, tempstep, dbase


# def check_multilayer(dbname, idSample):
#     """"Checks if the given sample is a multilayer sample"""

#     global samplesRecord
#     if samplesRecord is None:
#         db = sqlite3.connect(dbname)
#         samplesRecord = pd.read_sql_query("""SELECT IdSample, COUNT(*) As LayerCount
#                                             FROM SamplesLayers
#                                             GROUP BY IdSample""", db)
#         db.close()

#     rec = samplesRecord[samplesRecord["IdSample"] == idSample]
#     isMultilayer = rec["LayerCount"][0] > 1

#     return isMultilayer


def get_layerdata(dbname, idSample, unknownLayerId, layerData, materialProperties):
    """"Gets data of layers for a specific sample"""
    layers = []
    layerDataForSample = layerData[layerData["IdSample"] == idSample]
    for index, curLayerData in layerDataForSample.iterrows():
        # Get Layer name from index
        layerName = "middle"
        if curLayerData["layerIndex"] == 0:
            layerName = "bottom"
        elif curLayerData["layerIndex"] == (layerDataForSample["layerIndex"].count() - 1):
            layerName = "top"
        propertiesDf = materialProperties[
            materialProperties["IdMaterial"] == curLayerData["IdMaterial"]]
        propertiesDf = propertiesDf.reset_index()
        isUnknown = curLayerData["layerIndex"] == unknownLayerId
        layer = LayerData(curLayerData["IdSample"], curLayerData["layerIndex"],
                          layerName, curLayerData["Thickness"],
                          curLayerData["Name"], curLayerData["RefTemp"],
                          curLayerData["RefTempDensity"], isUnknown, propertiesDf)
        layers.append(layer)
    return layers


def read_database(db, name, plotswitch):
    """Read sqlite database and extract useful information"""



    # read tables from database into dataframes
    ids = pd.read_sql_query("SELECT * FROM ShotsPoints;", db)
    shots = pd.read_sql_query("SELECT * FROM Shots", db)
    substeps = pd.read_sql_query("SELECT * FROM SubSteps", db)
    tempsteps = pd.read_sql_query("SELECT * FROM TempSteps", db)
    mments = pd.read_sql_query("SELECT * FROM Measurements", db)
    mments_467 = pd.read_sql_query("SELECT * FROM Measurements_Lfa467", db)
    samples = pd.read_sql_query("SELECT * FROM Samples", db)
    sampleslayers = pd.read_sql_query("SELECT * FROM SamplesLayers", db)
    materials = pd.read_sql_query("SELECT * FROM Materials", db)
    mattoprop = pd.read_sql_query("SELECT * FROM MaterialToProperties", db)
    matprop = pd.read_sql_query("SELECT * FROM MaterialsProperties", db)

    # this is for the multilayer check
    samplesRecord = pd.read_sql_query("""SELECT IdSample, COUNT(*) As LayerCount
                                            FROM SamplesLayers
                                            GROUP BY IdSample""", db)
    layerData = pd.read_sql_query(
        """SELECT SamplesLayers.IdSample, SamplesLayers.IdMaterial,
        SamplesLayers.Ordinal as layerIndex, SamplesLayers.Thickness,
        Materials.Name, Materials.RefTemp, Materials.RefTempDensity
        FROM SamplesLayers
        INNER JOIN Materials
        ON SamplesLayers.IdMaterial = Materials.Id""", db)

    materialProperties = pd.read_sql_query(
        """SELECT MaterialToProperties.IdMaterial, MaterialsProperties.Type,
        MaterialsProperties.FileName, MaterialsProperties.Points
        FROM MaterialToProperties
        INNER JOIN MaterialsProperties
        ON MaterialToProperties.IdProperties = MaterialsProperties.Id""", db)

    db.close()

    # remove redundant column from dataframe
    matprop.drop(columns=['Type', 'DateTime'], inplace=True)
    # rename type in mattoprop due to same column name in ShotsPoints
    mattoprop.rename(columns={'Type': 'Type_materials'}, inplace=True)

    # merge dataframes and cleaning up
    merge = pd.merge(ids, shots, left_on="IdShot", right_on="Id")
    merge = merge[['DateTime', 'Data', 'IdShot', 'Type', 'Temperature', 'IdSubstep']]
    merge = pd.merge(merge, substeps, left_on="IdSubstep", right_on="Id")
    merge = merge[['DateTime', 'Data', 'IdShot', 'Type', 'Temperature', 'IdTempStep']]
    merge = pd.merge(merge, tempsteps, left_on="IdTempStep", right_on="Id")
    merge = merge[['DateTime', 'Data', 'IdShot', 'Type', 'Temperature', 'IdMment',
                   'FinalTemperature']]
    merge = pd.merge(merge, mments, left_on="IdMment", right_on="Id")
    merge = merge[['DateTime', 'Data', 'IdShot', 'Type', 'Temperature', 'IdSample', "IdMment",
                   'FinalTemperature']]

    # device specific measurements -->
    merge = pd.merge(merge, mments_467, left_on="IdMment", right_on="MmentId")
    merge = merge[['DateTime', 'Data', 'IdShot', 'Type', 'Temperature', 'IdSample',
                   'FinalTemperature', 'SpotSize']]
    # <--- device specic measurements END

    merge = pd.merge(merge, samples, left_on="IdSample", right_on="Id")
    merge = merge[['DateTime', 'Data', 'IdShot', 'Type', 'Temperature', 'Name',
                   'Diameter', 'Dim1', 'Dim2', 'Dim3', 'Length', 'IdSample',
                   'FinalTemperature', 'SpotSize', 'UnknownLayer']]
    merge = pd.merge(merge, sampleslayers, left_on="IdSample",
                     right_on="IdSample")
    merge = merge[['DateTime', 'Data', 'IdShot', 'Type', 'Temperature', 'Name',
                   'Diameter', 'Dim1', 'Dim2', 'Dim3', 'Length', 'IdMaterial',
                   'FinalTemperature', 'SpotSize', 'IdSample', 'UnknownLayer']]
    materials.rename(columns={'Name': 'Name_verbose'}, inplace=True)
    merge = pd.merge(merge, materials, left_on="IdMaterial", right_on="Id")
    merge = merge[['DateTime', 'Data', 'IdShot', 'Type', 'Temperature', 'Name',
                   'Diameter', 'Dim1', 'Dim2', 'Dim3', 'Length', 'RefTemp',
                   'RefTempDensity', 'Name_verbose', 'IdMaterial',
                   'FinalTemperature', 'SpotSize', 'IdSample', 'UnknownLayer']]
    merge = pd.merge(merge, mattoprop, left_on="IdMaterial",
                     right_on="IdMaterial")
    merge = merge[['DateTime', 'Data', 'IdShot', 'Type', 'Temperature', 'Name',
                   'Diameter', 'Dim1', 'Dim2', 'Dim3', 'Length', 'RefTemp',
                   'RefTempDensity', 'Name_verbose', 'Type_materials',
                   'IdProperties', 'FinalTemperature', 'SpotSize', 'IdSample', 'UnknownLayer']]
    merge = pd.merge(merge, matprop, left_on="IdProperties", right_on="Id")
    df = merge[['DateTime', 'Data', 'Diameter', 'Dim1', 'Dim2', 'Dim3',
                'IdShot', 'Length', 'Name', 'Points', 'RefTemp',
                'RefTempDensity', 'Temperature', 'Type', 'Type_materials',
                'FinalTemperature', 'SpotSize', 'IdSample', 'UnknownLayer']]

    del merge, ids, shots, substeps, tempsteps, mments, samples,\
        sampleslayers, materials, mattoprop, matprop

    # create list of all IdShot values and eleminate duplicates
    # duplicates come from separate entry of pulse and detector
    idshot_list = []
    for entry in df.IdShot:
        idshot_list.append(entry)
    idshot_list = list(dict.fromkeys(idshot_list))

    # parse all shots, extract detector and pulse signal and read material and temperature
    # save all this to class object "measurement"
    # create list of all measurements
    measurements = []
    for idshot in idshot_list:
        quadruplet = df[df["IdShot"] == idshot]

        # read data from detector and pulse
        # Type = 1 marks the data from the detector
        entry = quadruplet[quadruplet["Type"] == 1]
        ptime, pulse = convert_data(entry.Data.values[0], 'f')
        # Type = 2 marks the data from the pulse
        entry2 = quadruplet[quadruplet["Type"] == 2]
        dtime, detector = convert_data(entry2.Data.values[0], 'f')

        # metadata
        idSample = entry.IdSample.values[0]
        unknownLayer = entry.UnknownLayer.values[0]
        fintemp = entry.FinalTemperature.values[0]
        material = entry.Name.values[0]
        temp = entry.Temperature.values[0]
        diameter = entry.Diameter.values[0]
        d0 = entry.Dim1.values[0]
        d1 = entry.Dim2.values[0]
        d2 = entry.Dim3.values[0]
        thickness_rt = entry.Length.values[0]
        datetime = entry.DateTime.values[0]
        spotsize = entry.SpotSize.values[0]
        scl = bspline(entry.Points.values[0], temp)      # this is the expansion
        thickness_t = calc_thickness(thickness_rt, scl)
        # specific heat capacity
        cp_t = bspline(quadruplet[quadruplet["Type_materials"] == 1].Points.values[0], temp)
        if cp_t == 0:
            cp_t = 1

        rho_tref = entry.RefTempDensity.values[0]
        temp_ref = entry.RefTemp.values[0]
        alpha_tref = bspline(entry.Points.values[0], temp_ref)
        alpha_t = bspline(entry.Points.values[0], temp)
        rho_t = calc_density(rho_tref, alpha_tref, alpha_t)
        thermal_diff = bspline(quadruplet[quadruplet["Type_materials"] == 2].Points.values[0],
                               temp)
        if thermal_diff == 0:
            thermal_diff = 1

        # check whether measurement is multilayer
        rec = samplesRecord[samplesRecord["IdSample"] == idSample]
        try:
            isMultilayer = rec["LayerCount"][0] > 1
        except KeyError:
            isMultilayer = False

        # read multilayer data
        if(isMultilayer):
            layerData = get_layerdata(name, idSample, unknownLayer, layerData,
                                      materialProperties)

        # store data in class
        mment = Data(name, dtime, detector, ptime, pulse, material, temp,
                     diameter, d0, d1, d2, thickness_t, thickness_rt,
                     cp_t, rho_t, thermal_diff, datetime, fintemp, spotsize,
                     isMultilayer, layerData)

        if plotswitch is True:
            plot_shot(detector, dtime, pulse, ptime, material, temp, datetime)
        measurements.append(mment)
    for mment in measurements:
        write_shotfile(mment)


def plot_shot(detector, dtime, pulse, ptime, material, temperature, datetime):
    """Plot detector signal and pulse signal."""
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, sharey=True)
    ax1.plot(ptime, pulse)
    ax1.set_title('pulse')
    ax1.set_xlabel('time / ms')
    ax1.set_ylabel('signal / V')

    ax2.plot(dtime, detector)
    ax2.set_title('detector')
    ax2.set_xlabel('time / ms')
    ax1.set_ylabel('signal / V')

    tit = "{0} {1:.2f} °C".format(material, temperature)
    fig.suptitle(tit, fontsize=16)
    figname = material + " " + str(datetime) + ".png"
    plt.savefig(FIGPATH + figname)
    plt.close(fig)


def write_result(material, diffusivity, std_dev, temperature, outputname, tempstep, dbname):
    """Write result to csv file"""
    metadata = []
    metadata.append(['#Database', str(dbname)])
    metadata.append(['##General_information'])
    metadata.append(['#Material', str(material)])
    metadata.append(['#Tempstep', str(tempstep)])
    metadata.append(['##Results'])
    metadata.append(['#Shot number', '#Temperature/°C',
                     '#Diffusivity/(mm^2/s)', '#Std_Dev/(mm^2/s)',
                     '#Uncertainty/%'])
    data = []
    uncertainty = std_dev / diffusivity * 100

    # change localisation of numbers, to match existing format
    # temperature = format_decimal(temperature, locale='de_DE',
    #                              decimal_quantization=False)
    # diffusivity = format_decimal(diffusivity, locale='de_DE',
    #                              decimal_quantization=False)
    # std_dev = format_decimal(std_dev, locale='de_DE',
    #                          decimal_quantization=False)
    # uncertainty = format_decimal(uncertainty, locale='de_DE',
    #                              decimal_quantization=False)
    data.append(['1...', temperature, diffusivity, std_dev, uncertainty])

    # write detector data to csv file
    with open(outputname, 'w', encoding='UTF8') as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerows(metadata)
        writer.writerows(data)


def write_shotfile(mment):
    """Write CSV files for detector and pulse for the given measurement"""
    # metadata
    metadata = []
    metadata.append(['##Shot_data', str(1)])
    metadata.append([])
    metadata.append(['#Database', str(mment.dbname)])
    metadata.append([])
    metadata.append(['##General_information'])
    # TODO: comment-in remaining information
    metadata.append(['#Diameter/mm', str(mment.diameter)])
    # metadata.append(['#FinalTemperature', str(mment.fintemp)])
    metadata.append(['#FinalTemperature', str(mment.temp)])
    metadata.append(['#Material', str(mment.material)])
    metadata.append(['#Spotsize', str(mment.spotsize)])
    metadata.append(['#Thickness_RT/mm', str(mment.thickness_rt)])
    metadata.append([])

    # multilayer data
    if mment.isMultilayer:
        logging.info("Writing multilayer measurement {0}".format(str(mment.material)))
        line_layer = ['#Layer']
        line_name = ['#Name']
        line_material = ['#Material']
        line_refTemp = ['#Ref_temperature /°C']
        line_refTempDens = ['#Ref_density /(g/cm^3)']
        line_cpTable = ['#Cp_table']
        line_thermalExpansion = ['#Thermal_expansion_table']
        line_thermalDiff = ['#Thermal_diffusivity_table']
        line_cpT = ['#Cp_T/(J/(g*K))']
        line_density = ['#Density_T/(g/cm^3)']
        line_diff = ['#Thermal_diffusivity_T/(mm^2/s)']
        line_thickness = ['#Thickness_T/mm']

        for curLayer in mment.layerData:
            if curLayer.isUnknown:
                continue

            # get cp
            pointsType1 = curLayer.matProperties[
                curLayer.matProperties["Type"] == 1]["Points"].values[0]
            cp_t = bspline(pointsType1, curLayer.refTemp)

            # get density
            alpha_tref = bspline(pointsType1, curLayer.refTemp)
            alpha_t = bspline(pointsType1, mment.temp)
            density = calc_density(curLayer.refTempDensity, alpha_tref, alpha_t)

            # get thermal diffusivity
            diff = bspline(curLayer.matProperties[
                curLayer.matProperties["Type"] == 2]["Points"].values[0], curLayer.refTemp)

            line_layer.append(str(curLayer.layerIndex + 1))
            line_name.append("#{0}".format(curLayer.layerName))
            line_material.append(curLayer.materialName)
            line_refTemp.append(curLayer.refTemp)
            line_refTempDens.append(curLayer.refTempDensity)
            line_cpTable.append(curLayer.matProperties[
                curLayer.matProperties["Type"] == 1]["FileName"].values[0])
            line_thermalExpansion.append(curLayer.matProperties[
                curLayer.matProperties["Type"] == 3]["FileName"].values[0])
            line_thermalDiff.append(curLayer.matProperties[
                curLayer.matProperties["Type"] == 2]["FileName"].values[0])
            line_cpT.append(cp_t)
            line_density.append(density)
            line_diff.append(diff)
            line_thickness.append(curLayer.thickness)

        metadata.append(['##Known_layers'])
        metadata.append(line_layer)
        metadata.append(line_name)
        metadata.append(line_material)
        metadata.append(line_refTemp)
        metadata.append(line_refTempDens)
        metadata.append(line_cpTable)
        metadata.append(line_thermalExpansion)
        metadata.append(line_thermalDiff)
        metadata.append(line_cpT)
        metadata.append(line_density)
        metadata.append(line_diff)
        metadata.append(line_thickness)
        metadata.append([])

    metadata.append(['##Shot_information'])
    # axial diffusivity uses the same value as diffusivity, but is only needed
    # for in-plane calculations. For multi-layer calculations, the same value
    # is used for the known layers
    metadata.append(['#Axial_diffus_T/(mm^2/s)', str(mment.thermal_diff)])
    metadata.append(['#Density_T/(g/cm^3)', str(mment.density_t)])
    metadata.append(['#Cp_T/(J/(g*K))', str(mment.cp_t)])
    metadata.append(['#D0/mm', str(mment.d0)])
    metadata.append(['#D1/mm', str(mment.d1)])
    metadata.append(['#D2/mm', str(mment.d2)])
    metadata.append(['#Sample_temperature/°C', str(mment.temp)])
    metadata.append(['#Thermal_diffusivity_T/(mm^2/s)', str(mment.thermal_diff)])
    metadata.append(['#Thickness_T/mm', str(mment.thickness_t)])
    metadata.append([])

    # detector data
    detector = []
    detector.append(['##Detector_data'])
    detector.append(['#Time/ms', '#Detector/V'])
    detectordata = zip(mment.dtime, mment.detector)

    # pulse data
    pulse = []
    pulse.append(['##Laser_pulse_data'])
    pulse.append(['#Time/ms', '#Pulse/V'])
    pulsedata = zip(mment.ptime, mment.pulse)

    # create path and name for each file and write data into files
    materialspaceless = mment.material.replace(" ", "")       # remove spaces
    materialspaceless = materialspaceless.replace("/", "")  # remove '/'
    # use MD5 sum of material name to avoid weird characters in filename
    uniqdirname = hashlib.md5(materialspaceless.encode('utf-8')).hexdigest()
    uniqdirpath = DATAPATH + uniqdirname
    subdirname = str(mment.fintemp)
    subdirpath = uniqdirpath + '/' + subdirname
    # create directory for material and subdirectory for FinalTemperature
    create_subdirs((uniqdirpath, subdirpath), False, None)

    detectorname = subdirpath + '/' + materialspaceless + "_" + mment.datetime\
        + '_detector.csv'
    pulsename = subdirpath + '/' + materialspaceless + "_" + mment.datetime\
        + '_pulse.csv'

    # write detector data to csv file
    with open(detectorname, 'w', encoding='UTF8') as f:
        writer = csv.writer(f, dialect="unix", quoting=csv.QUOTE_MINIMAL)
        writer.writerows(metadata)
        writer.writerows(detector)
        writer.writerows(detectordata)

    # write pulse data to csv file
    with open(pulsename, 'w', encoding='UTF8') as f:
        writer = csv.writer(f, dialect="unix", quoting=csv.QUOTE_MINIMAL)
        writer.writerows(metadata)
        writer.writerows(pulse)
        writer.writerows(pulsedata)


if __name__ == '__main__':



    # strip parameters and save in variables
    parser = argparse.ArgumentParser()

    parser.add_argument("--analyze", default=False, action="store_true")
    parser.add_argument("--baseline", type=int, default=1, choices=range(5))
    parser.add_argument("--calc", default=False, action="store_true")
    parser.add_argument("--database", default=None, action="store")
    parser.add_argument("--calcref", default=False, action="store_true")
    parser.add_argument("--clean", default=False, action="store_true")
    parser.add_argument("--code", type=int, default=0, choices=[0, 1])
    parser.add_argument("--compall", default=None, action="store_true")
    parser.add_argument("-m", "--mono", default=False, action="store_true")
    parser.add_argument("--res", default=False, action="store_true")
    parser.add_argument("--model", type=int, default=0,
                        choices=range(13))
    # TODO: Add subparser for --database to create plots of measured signals
    parser.add_argument("--plot", default=False, action="store_true")
    parser.add_argument("--pulse", type=int, default=1, choices=range(4))
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = parser.parse_args()


    logging.basicConfig(filename=LOGFILE,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        encoding='utf-8', level=logging.DEBUG, filemode='w')

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    ui.set_up_connection()
    MainWindow.show()

    sys.exit(app.exec_())

