import random
import math

from PySide2.QtGui import QColor, qGray


class ColorManager:

    @staticmethod
    def GetRandomColor():
        if not getattr(ColorManager.GetRandomColor, 'currentHue', None):
            ColorManager.GetRandomColor.currentHue = random.random()

        color = QColor.fromHslF(ColorManager.GetRandomColor.currentHue, 1.0, 0.5)
    
        ColorManager.GetRandomColor.currentHue = ColorManager.GetRandomColor.currentHue + 0.618033988749895
        ColorManager.GetRandomColor.currentHue = math.fmod(ColorManager.GetRandomColor.currentHue, 1.0)

        return color

    @staticmethod
    def AreColorsClose(color, anotherColor, threshold = 100):
        color        = QColor(color)
        anotherColor = QColor(anotherColor)

        redDifference   = color.red()   - anotherColor.red()
        blueDifference  = color.blue()  - anotherColor.blue()
        greenDifference = color.green() - anotherColor.green()

        return (redDifference ** 2 + blueDifference ** 2 + greenDifference ** 2) <= threshold * threshold

    @staticmethod
    def GetRandomColors(colorsNumber):
        return [ColorManager.GetRandomColor() for _ in range(colorsNumber)]

    @staticmethod
    def ToGrayScale(color):
        return QColor(qGray(QColor(color).rgb()), qGray(QColor(color).rgb()), qGray(QColor(color).rgb()))
