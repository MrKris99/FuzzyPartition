import sys
import functools
import random

from PySide2.QtWidgets import (QApplication, QPushButton, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QGridLayout, QLabel, QLineEdit, QComboBox, QSplitter, QCheckBox, QSpacerItem, QSizePolicy)
from PySide2.QtCore import Slot, Qt

from utils.cell import Cell
from utils.color_manager import ColorManager

from partition.partition_utils import (EuclidianDistance, ChebyshevDistance, TaxicabDistance, sign, 
    MonotonicStepFunction, ConstantStepFunction, SeriesStepFunction, LeftRectangularIntergate)
from partition.partition import ( FuzzyPartitionWithFixedCentersAlgorithm, 
SimplePartitionWithFixedCentersAlgorithm, FuzzyPartitionWithNotFixedCentersAlgorithm)


WINDOW_SIZE = 500


class SettingsWidget(QWidget):

    def __init__(self, boardWidget, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._boardWidget      = boardWidget
        self._freeCoefficients = dict()

        self._layout = self._buildSettingsLayout()
        self._layout.setAlignment(Qt.AlignTop)

        self.setLayout(self._layout)

    @Slot()
    def _onAddCenterButtonClicked(self):
        if not self._xCoordinateInput.text() or not self._yCoordinateInput.text():
            return

        x = round(float(self._xCoordinateInput.text()) * WINDOW_SIZE)
        y = round(float(self._yCoordinateInput.text()) * WINDOW_SIZE)

        wasAdded = self._boardWidget.addCenter(x, y)

        if not wasAdded:
            return

        freeCoeficientsLabel = QLabel(text='a<sub>i</sub> ({:3.3f}, {:3.3f}): '.format(
            x / WINDOW_SIZE, y / WINDOW_SIZE))

        self._freeCoeficientsInput = QLineEdit()
        self._freeCoeficientsInput.setPlaceholderText('0')

        freeCoeficientsLayout = QHBoxLayout()
        freeCoeficientsLayout.addWidget(freeCoeficientsLabel)
        freeCoeficientsLayout.addWidget(self._freeCoeficientsInput)

        self._layout.addLayout(freeCoeficientsLayout)

        self._freeCoefficients[(x, y)] = self._freeCoeficientsInput

    @Slot()
    def _activateFuzzyInput(self, newText):
        allControls = [
            self._confidenceInput, 
            self._precisionInput, 
            self._xCoordinateInput,
            self._yCoordinateInput,
            self._addCenterButton,
            self._centersCountInput
        ]

        activator = {
            'simple partition': [self._xCoordinateInput, self._yCoordinateInput, self._addCenterButton],
            'fuzzy partition (find centers)': [self._confidenceInput, self._precisionInput, self._centersCountInput],
            'fuzzy partition': [self._confidenceInput, 
            self._precisionInput, 
            self._xCoordinateInput,
            self._yCoordinateInput,
            self._addCenterButton,]
        }

        for item in activator:
            if item == newText.lower():
                for control in allControls:
                    control.setEnabled(control in activator[item])

    def _buildSettingsLayout(self):

        #------------------------------------Gray scale------------------------------------#

        grayScaleLabel          = QLabel(text='Gray color')
        self._grayScaleCheckbox = QCheckBox()

        grayScaleLayout = QHBoxLayout()
        grayScaleLayout.addWidget(grayScaleLabel)
        grayScaleLayout.addWidget(self._grayScaleCheckbox)

        grayScaleLayout.setAlignment(Qt.AlignRight)

        self._grayScaleCheckbox.stateChanged.connect(self._boardWidget.toGrayScale)

        #-----------------------------------------------------------------------------------#

        #------------------------------------Distance options------------------------------------#

        self._distanceOptions = QComboBox()
        self._distanceOptions.addItems(['Euclidian Distance', 'Chebyshev Distance', 'Taxicab Distance'])

        distanceLayout = QVBoxLayout()
        distanceLayout.addWidget(self._distanceOptions)

        #----------------------------------------------------------------------------------------#

        #------------------------------------Partition options------------------------------------#

        self._partitionOptions = QComboBox()
        self._partitionOptions.addItems(['Simple Partition', 'Fuzzy Partition', 'Fuzzy Partition (find centers)'])

        partitionLayout = QVBoxLayout()
        partitionLayout.addWidget(self._partitionOptions)

        self._partitionOptions.currentTextChanged.connect(self._activateFuzzyInput)

        #-----------------------------------------------------------------------------------------#

        #------------------------------------Confidence Degree------------------------------------#

        self._confidenceInput = QLineEdit(enabled=False)
        self._confidenceInput.setPlaceholderText('Confidence degree')

        confidenceLayout = QVBoxLayout()
        confidenceLayout.addWidget(self._confidenceInput)

        self._precisionInput = QLineEdit(enabled=False)
        self._precisionInput.setPlaceholderText('Gradient method precision')

        precisionLayout = QVBoxLayout()
        precisionLayout.addWidget(self._precisionInput)

        self._centersCountInput = QLineEdit(enabled=False)
        self._centersCountInput.setPlaceholderText('Centers number')

        centersCountLayout = QVBoxLayout()
        centersCountLayout.addWidget(self._centersCountInput)

        #-----------------------------------------------------------------------------------------#

        #------------------------------------Add center------------------------------------#

        self._xCoordinateInput = QLineEdit()
        self._xCoordinateInput.setPlaceholderText('x center coordinate')

        self._yCoordinateInput = QLineEdit()
        self._yCoordinateInput.setPlaceholderText('y center coordinate')

        self._addCenterButton = QPushButton()
        self._addCenterButton.setText('Add new center')
        self._addCenterButton.setMinimumWidth(400)

        addCentersLayout = QVBoxLayout()
        addCentersLayout.addWidget(self._xCoordinateInput)
        addCentersLayout.addWidget(self._yCoordinateInput)
        addCentersLayout.addWidget(self._addCenterButton)

        self._addCenterButton.clicked.connect(self._onAddCenterButtonClicked)

        #-----------------------------------------------------------------------------------#

        spacer = QSpacerItem(40, 40, QSizePolicy.Expanding, QSizePolicy.Minimum)

        settingsLayout = QVBoxLayout()
        settingsLayout.addLayout(grayScaleLayout)
        settingsLayout.addLayout(partitionLayout)
        settingsLayout.addLayout(distanceLayout)
        settingsLayout.addLayout(confidenceLayout)
        settingsLayout.addLayout(precisionLayout)
        settingsLayout.addLayout(centersCountLayout)
        settingsLayout.addItem(spacer)
        settingsLayout.addLayout(addCentersLayout)

        return settingsLayout

    def distance(self):
        returnDistance = EuclidianDistance

        if 'euclidian' in self._distanceOptions.currentText().lower():
            returnDistance = EuclidianDistance
        elif 'cheb' in self._distanceOptions.currentText().lower():
            returnDistance = ChebyshevDistance
        elif 'taxi' in self._distanceOptions.currentText().lower():
            returnDistance = TaxicabDistance

        return returnDistance

    def partitionAlgorithm(self):
        return self._partitionOptions.currentText().lower()

    def precision(self):
        return float(self._precisionInput.text()) if self._precisionInput.text() else 0.01

    def confidence(self):
        return float(self._confidenceInput.text()) if self._confidenceInput.text() else 0.0

    def freeCoefficients(self):
        return { center: float(freeCoefficient.text()) * WINDOW_SIZE
            if freeCoefficient.text() else 0.0 for center, freeCoefficient in self._freeCoefficients.items() }

    def centersNumber(self):
        return int(self._centersCountInput.text()) if self._centersCountInput.text() else 1

    def grayScale(self):
        return self._grayScaleCheckbox.isChecked()

class PartitionCentralWidget(QWidget):

    def __init__(self, boardWidget, settingsWidget, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._boardWidget    = boardWidget
        self._settingsWidget = settingsWidget

        self._startPartitionButton = QPushButton('Start Partition')
        self._startPartitionButton.clicked.connect(self._onStartPartitionButtonClicked)

        self._splitter = QSplitter(Qt.Horizontal)
        self._splitter.addWidget(self._boardWidget)
        self._splitter.addWidget(self._settingsWidget)

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self._splitter)
        mainLayout.addWidget(self._startPartitionButton)

        self.setLayout(mainLayout)

    @Slot()
    def _onStartPartitionButtonClicked(self):
        partitionAlgorithm = self._settingsWidget.partitionAlgorithm()

        if 'simple' in partitionAlgorithm.lower():
            self._boardWidget.startSimplePartition(
                self._settingsWidget.distance(), self._settingsWidget.freeCoefficients(), self._settingsWidget.grayScale())
        elif 'centers' in partitionAlgorithm.lower():
            self._boardWidget.startFuzzyPartitionNotFixedCenters(
                self._settingsWidget.distance(), 
                self._settingsWidget.confidence(), 
                self._settingsWidget.centersNumber(),
                self._settingsWidget.precision(),
                self._settingsWidget.grayScale())
        else:
            self._boardWidget.startFuzzyPartition(
                self._settingsWidget.distance(), 
                self._settingsWidget.confidence(), 
                self._settingsWidget.freeCoefficients(),
                self._settingsWidget.precision(),
                self._settingsWidget.grayScale())


class BoardWidget(QWidget):

    CELL_CENTER_COLOR = Qt.black
    CELL_SIMPLE_COLOR = Qt.white

    def __init__(self, size, cellSize, application, statusBarWidget, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._application     = application
        self._statusBarWidget = statusBarWidget
        self._windowSize      = size
        self._cellSize        = cellSize
        self._board           = dict()

        for cellX in range(0, self._windowSize, self._cellSize):
            for cellY in range(0, self._windowSize, self._cellSize):
                self._board[(cellX, cellY)] = Cell(cellX, cellY, color=self.CELL_SIMPLE_COLOR, parent=self)
                self._board[(cellX, cellY)].resize(self._cellSize, self._cellSize)

        self.setFixedSize(self._windowSize, self._windowSize)

    def mouseEnterEvent(self, cell):
        newX, newY = cell.x() / self._windowSize, cell.y() / self._windowSize
        
        self._statusBarWidget.showMessage('x: {}, y: {}'.format(
            round(newX, 3), round(newY, 3)))

    def mouseLeaveEvent(self, cell):
        self._statusBarWidget.showMessage('')

    def addCenter(self, x, y):
        if (x, y) in self._board.keys() and not self._board[(x, y)].isCenter:
            self._toogleCenter(self._board[(x, y)])
            return True

        return False

    def toGrayScale(self, state):
        for point in self._board.values():
            if not point.isCenter:
                if state == Qt.Checked:
                    point.previousColor = point.color
                    point.color         = ColorManager.ToGrayScale(point.color) 
                else:
                    point.color = point.previousColor

    def startSimplePartition(self, distance, freeCoefficients, grayScale=False):
        self._clearBoard()

        centers = [
            point for point in self._board.keys() if self._board[point].isCenter
        ]

        board = [
            point for point in self._board.keys()
        ]

        partition = SimplePartitionWithFixedCentersAlgorithm(
            board, centers, freeCoefficients)

        colors = dict(zip(centers, ColorManager.GetRandomColors(len(centers))))

        for point, center in partition.calculatePartition(distance):
            if center:
                self._board[point].color = ColorManager.ToGrayScale(colors[center]) if grayScale else colors[center]

            self._application.processEvents()

    def startFuzzyPartition(self, distance, confidenceDeegre, freeCoefficients, precision, grayScale=False):
        self._clearBoard()

        centers = [
            point for point in self._board.keys() if self._board[point].isCenter
        ]

        board = [
            point for point in self._board.keys()
        ]

        partition = FuzzyPartitionWithFixedCentersAlgorithm(
            board, centers, SeriesStepFunction(25), confidenceDeegre, freeCoefficients, precision)

        colors = dict(zip(centers, ColorManager.GetRandomColors(len(centers))))
    
        for point, center, _ in partition.calculatePartition(distance):
            if center:
                self._board[point].color = ColorManager.ToGrayScale(colors[center]) if grayScale else colors[center]

            self._application.processEvents()

    def startFuzzyPartitionNotFixedCenters(self, distance, confidenceDeegre, centersCount, precision, grayScale=False):
        self._clearBoard()

        board = [
            point for point in self._board.keys()
        ]

        startCenters = [(100, 100)] * centersCount

        for point in startCenters:
            self._board[point].color = Qt.red
            self._application.processEvents()

        partition = FuzzyPartitionWithNotFixedCentersAlgorithm(
            board, centersCount, SeriesStepFunction(25), confidenceDeegre, None, precision)

        for newBoard in partition.calculatePartition(distance, startCenters):
            self._clearBoard()

            centers = set()

            for point in newBoard:
                if newBoard[point].center:
                    centers.add(newBoard[point].center)

            centers = list(centers)
            colors  = dict(zip(centers, ColorManager.GetRandomColors(len(centers))))

            for point in newBoard:
                if newBoard[point].center:
                    self._board[point].color = colors[newBoard[point].center]

                    if point in centers:
                        self._board[point].color = Qt.red

                self._application.processEvents()

    def _toogleCenter(self, cell):
        cell.isCenter = not cell.isCenter
        cell.color    = self.CELL_CENTER_COLOR if cell.isCenter else self.CELL_SIMPLE_COLOR

    def _clearBoard(self):
        for cell in self._board.values():
            if not cell.isCenter:
                cell.color = self.CELL_SIMPLE_COLOR

    
if __name__ == '__main__':
    application = QApplication(sys.argv)

    mainWindow = QMainWindow()
    mainWindow.setWindowTitle('Partition')

    boardWidget = BoardWidget(WINDOW_SIZE, 5, application, mainWindow.statusBar())

    settingsWidget = SettingsWidget(boardWidget)

    mainWindow.setCentralWidget(PartitionCentralWidget(boardWidget, settingsWidget))
    mainWindow.show()

    sys.exit(application.exec_())