import math
import functools
import abc


sign = functools.partial(math.copysign, 1)


def RoundNextMultiple(number, base=5):
    return base * round(number / base)


def LeftRectangularIntergate(function, a, b, c, d, xPointsNumber, yPointsNumber):
	stepX = (b - a) / xPointsNumber
	stepY = (d - c) / yPointsNumber

	integral = 0

	for i in range(xPointsNumber):
		for j in range(yPointsNumber):
			nextX = a + i * stepX
			nextY = c + j * stepY

			integral += stepX * stepY * function(nextX, nextY)

	return integral


def EuclidianDistance(point, otherPoint):
    return math.sqrt(sum([math.pow((a - b), 2) for a, b in zip(point, otherPoint)]))


def ChebyshevDistance(point, otherPoint):
    return max([abs(a - b) for a, b in zip(point, otherPoint)])


def TaxicabDistance(point, otherPoint):
    return sum([abs(a - b) for a, b in zip(point, otherPoint)])


class StepFunction:

    @abc.abstractmethod
    def getNextStep(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass


class ConstantStepFunction(StepFunction):

    def __init__(self, constantStep):
        super().__init__()

        self._constantStep = constantStep

    def getNextStep(self):
        return self._constantStep

    def reset(self):
        pass


class MonotonicStepFunction(StepFunction):

    def getNextStep(self):
        return 5

    def reset(self):
        pass


class SeriesStepFunction(StepFunction):
    
    def __init__(self, constant):
        super().__init__()

        self._constant    = constant
        self._nextElement = 0

    def getNextStep(self):
        self._nextElement += 1

        return self._constant / (math.log(self._nextElement) + 1)

    def reset(self):
        self._nextElement = 0