from PySide2.QtWidgets import QPushButton
from PySide2.QtGui import QPalette


class Cell(QPushButton):
    
    def __init__(self, x, y, color, isCenter=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.isCenter    = isCenter
        self.color       = color

        self.move(x, y)

    def getIsCenter(self):
        return self._isCenter

    def setIsCenter(self, isCenter):
        self._isCenter = isCenter

    isCenter = property(getIsCenter, setIsCenter)

    def getColor(self):
        return self._color

    def setColor(self, newColor):
        self._color = newColor

        palette = QPalette(self.palette())

        palette.setColor(QPalette.Button    , self.color)
        palette.setColor(QPalette.Background, self.color)

        self.setAutoFillBackground(True)
        self.setPalette(palette)

    color = property(getColor, setColor)

    def enterEvent(self, event):
        self.parent().mouseEnterEvent(self)

    def leaveEvent(self, event):
        self.parent().mouseLeaveEvent(self)