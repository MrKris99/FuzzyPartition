import sys
from threading import Thread

from PySide2.QtWidgets import (QApplication, QPushButton, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QLineEdit, QComboBox, QSplitter, QSpacerItem, QSizePolicy, QGraphicsView, QGraphicsScene)
from PySide2.QtCore import Slot, Qt, QRectF, QRect, Signal
from PySide2.QtGui import QPen, QBrush

from utils.cell import Cell
from utils.color_manager import ColorManager

from partition.partition_utils import (EuclidianDistance, ChebyshevDistance, TaxicabDistance, sign, 
    MonotonicStepFunction, ConstantStepFunction, SeriesStepFunction, LeftRectangularIntergate)
from partition.partition import ( FuzzyPartitionWithFixedCentersAlgorithm, SimplePartitionWithFixedCentersAlgorithm)

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
        ]

        activator = {
            'simple partition': [self._xCoordinateInput, self._yCoordinateInput, self._addCenterButton],
            'fuzzy partition': [self._confidenceInput, self._precisionInput, self._xCoordinateInput, self._yCoordinateInput, self._addCenterButton,]
        }

        for item in activator:
            if item == newText.lower():
                for control in allControls:
                    control.setEnabled(control in activator[item])

    def _buildSettingsLayout(self):

        #------------------------------------Distance options------------------------------------#

        self._distanceOptions = QComboBox()
        self._distanceOptions.addItems(['Euclidian Distance', 'Chebyshev Distance', 'Taxicab Distance'])

        distanceLayout = QVBoxLayout()
        distanceLayout.addWidget(self._distanceOptions)

        #----------------------------------------------------------------------------------------#

        #------------------------------------Partition options------------------------------------#

        self._partitionOptions = QComboBox()
        self._partitionOptions.addItems(['Simple Partition', 'Fuzzy Partition'])

        partitionLayout = QVBoxLayout()
        partitionLayout.addWidget(self._partitionOptions)

        self._partitionOptions.currentTextChanged.connect(self._activateFuzzyInput)

        #-----------------------------------------------------------------------------------------#

        #------------------------------------Confidence Degree------------------------------------#

        self._confidenceInput = QLineEdit(enabled=True)
        self._confidenceInput.setPlaceholderText('Confidence degree')

        confidenceLayout = QVBoxLayout()
        confidenceLayout.addWidget(self._confidenceInput)

        self._precisionInput = QLineEdit(enabled=True)
        self._precisionInput.setPlaceholderText('Gradient method precision')

        precisionLayout = QVBoxLayout()
        precisionLayout.addWidget(self._precisionInput)

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
        settingsLayout.addLayout(partitionLayout)
        settingsLayout.addLayout(distanceLayout)
        settingsLayout.addLayout(confidenceLayout)
        settingsLayout.addLayout(precisionLayout)
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

class PartitionCentralWidget(QWidget):

    def __init__(self, boardWidget, settingsWidget, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._boardWidget    = boardWidget
        self._settingsWidget = settingsWidget

        self._startPartitionButton = QPushButton('Start Partition')
        self._paintGrayScale = QPushButton('Paint as grayscale')
        self._paintGrayScale.setEnabled(False)
        self._startPartitionButton.clicked.connect(self._onStartPartitionButtonClicked)
        self._paintGrayScale.clicked.connect(self._onPaintGrayScaleButtonClicked)

        self._splitter = QSplitter(Qt.Horizontal)
        self._splitter.addWidget(self._boardWidget)
        self._splitter.addWidget(self._settingsWidget)

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self._splitter)
        mainLayout.addWidget(self._startPartitionButton)
        mainLayout.addWidget(self._paintGrayScale)

        self.setLayout(mainLayout)

    @Slot()
    def _onStartPartitionButtonClicked(self):
        self._startPartitionButton.setEnabled(False)
        partitionAlgorithm = self._settingsWidget.partitionAlgorithm()

        if 'simple' in partitionAlgorithm.lower():
            self._boardWidget.startSimplePartition(
                self._settingsWidget.distance(), self._settingsWidget.freeCoefficients())
        else:
            self._boardWidget.startFuzzyPartition(
                self._settingsWidget.distance(), 
                self._settingsWidget.confidence(), 
                self._settingsWidget.freeCoefficients(),
                self._settingsWidget.precision())

        self._startPartitionButton.setEnabled(True)
        self._paintGrayScale.setEnabled(True)

    @Slot()
    def _onPaintGrayScaleButtonClicked(self):
        self._paintGrayScale.setEnabled(False)
        self._startPartitionButton.setEnabled(False)
        self._boardWidget.toGrayScale()
        self._startPartitionButton.setEnabled(True)

class BoardWidget(QGraphicsView):

    CELL_CENTER_COLOR = Qt.black
    CELL_SIMPLE_COLOR = Qt.white
    CELL_SIMPLE_GRAY_COLOR = Qt.gray
    CELL_BOUNDS_COLOR = Qt.black
    update_progress = Signal(object)

    def __init__(self, size, cellSize, application, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._application = application
        self._windowSize  = size
        self._cellSize    = cellSize
        self._board       = dict()
        self._colors      = dict()
        self._scene       = QGraphicsScene()

        self.setScene(self._scene)
        self.setFixedSize(self._windowSize * 1.05, self._windowSize * 1.05)
        self.update_progress.connect(self.updateProgress)

        for cellX in range(0, self._windowSize, self._cellSize):
            for cellY in range(0, self._windowSize, self._cellSize):
                self._board[(cellX, cellY)] = Cell(color=self.CELL_SIMPLE_COLOR)
                self.updateColorAt((cellX, cellY))

    def updateColorAt(self, point):
        rect = QRectF(point[0], point[1], self._cellSize, self._cellSize)
        self._scene.invalidate(rect)
        self._scene.addRect(rect, QPen(Qt.black), QBrush(self._board[point].color))

    def addCenter(self, x, y):
        if (x, y) in self._board.keys():
            cell = self._board[(x, y)]
            if cell.isCenter:
                return False
            cell.isCenter = True
            cell.color    = self.CELL_CENTER_COLOR
            self.updateColorAt((x, y))
            return True

        return False

    def getExtraneousAdjecements(self, x, y):
        adjecements = []
        if (x < self._cellSize or x + self._cellSize >= self._windowSize):
            return adjecements
        if (y < self._cellSize or y + self._cellSize >= self._windowSize):
            return adjecements
        if (self._board[(x, y)].color == self.CELL_BOUNDS_COLOR):
            return adjecements

        checkedColor = self._board[(x, y)].color
        for xDelta in [0, self._cellSize, -self._cellSize]:
            for yDelta in [0, self._cellSize, -self._cellSize]:
                candidate = self._board[(x + xDelta, y + yDelta)];
                if (candidate.previousColor != checkedColor and candidate.color != self.CELL_BOUNDS_COLOR):
                    adjecements.append((x + xDelta, y + yDelta))
        return adjecements

    def forEachPoint(self, callback):
        for x in range(0, self._windowSize, self._cellSize):
            for y in range(0, self._windowSize, self._cellSize):
                callback(x, y)

    def savePrevColorAndGrayifyNeutralCells(self, x, y):
        self._board[(x, y)].previousColor = self._board[(x, y)].color
        if (self._board[(x, y)].color == self.CELL_SIMPLE_COLOR):
            self._board[(x, y)].color = self.CELL_SIMPLE_GRAY_COLOR
            self.updateColorAt((x, y))

    def markBoundsAsBlack(self, x, y):
        adjecements = self.getExtraneousAdjecements(x, y)
        for point in adjecements:
            if (self._board[point].color != self.CELL_SIMPLE_GRAY_COLOR):
                self._board[point].color = self.CELL_BOUNDS_COLOR
                self.updateColorAt(point)

    def markRelatedPointAsWhite(self, x, y):
        if (self._board[(x, y)].color != self.CELL_SIMPLE_GRAY_COLOR and self._board[(x, y)].color != self.CELL_BOUNDS_COLOR):
            self._board[(x, y)].color = self.CELL_SIMPLE_COLOR
            self.updateColorAt((x, y))

    def toGrayScale(self):
        self.forEachPoint(self.savePrevColorAndGrayifyNeutralCells)
        self.forEachPoint(self.markBoundsAsBlack)
        self.forEachPoint(self.markRelatedPointAsWhite)

    @Slot(object)
    def updateProgress(self, object):
        point = object[0]
        center = object[1]
        if center:
            self._board[point].color = self._colors[center]
            self.updateColorAt(point)

    def pointCalculatedCallback(self, point, center):
        self.update_progress.emit((point, center))

    def startSimplePartition(self, distance, freeCoefficients):
        self._clearBoard()
        centers = [ point for point in self._board.keys() if self._board[point].isCenter]
        board = [ point for point in self._board.keys()]
        partition = SimplePartitionWithFixedCentersAlgorithm(
            board, centers, freeCoefficients, distance, self.pointCalculatedCallback)

        self._colors = dict(zip(centers, ColorManager.GetRandomColors(len(centers))))
        thread = Thread(target = partition.calculatePartition)
        thread.start()
 
    def startFuzzyPartition(self, distance, confidenceDeegre, freeCoefficients, precision):
        self._clearBoard()
        centers = [ point for point in self._board.keys() if self._board[point].isCenter]
        board = [ point for point in self._board.keys()]
        partition = FuzzyPartitionWithFixedCentersAlgorithm(
            board, centers, SeriesStepFunction(25), confidenceDeegre, freeCoefficients, distance, self.pointCalculatedCallback, precision)

        self._colors = dict(zip(centers, ColorManager.GetRandomColors(len(centers))))
        thread = Thread(target = partition.calculatePartition)
        thread.start()

    def _clearBoard(self):
        for point, cell in self._board.items():
            if not cell.isCenter:
                cell.color = self.CELL_SIMPLE_COLOR
                self.updateColorAt(point)

if __name__ == '__main__':
    application = QApplication(sys.argv)

    mainWindow = QMainWindow()
    mainWindow.setWindowTitle('Partition')

    boardWidget = BoardWidget(WINDOW_SIZE, 5, application)
    settingsWidget = SettingsWidget(boardWidget)
    centralWidget = PartitionCentralWidget(boardWidget, settingsWidget)

    mainWindow.setCentralWidget(centralWidget)
    mainWindow.show()

    sys.exit(application.exec_())
