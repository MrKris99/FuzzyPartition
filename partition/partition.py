import abc
import functools
from .partition_utils import sign

class BasePartitionAlgorithm:

    @abc.abstractmethod
    def calculatePartition(self, distance):
        pass


class BasePartitionWithFixedCentersAlgorithm(BasePartitionAlgorithm):

    def __init__(self, board, centers, freeCoefficients):
        super().__init__()

        self._board            = board
        self._centers          = centers
        self._freeCoefficients = freeCoefficients

    def calculatePartition(self, distance):
        points = list(point for point in self._board if point not in self._centers)
        for point in points:
            result = self._calculateForPoint(point, distance)
            yield result

    @abc.abstractmethod
    def _calculateForPoint(self, point, distance):
        pass


class SimplePartitionWithFixedCentersAlgorithm(BasePartitionWithFixedCentersAlgorithm):

    def __init__(self, board, centers, freeCoefficients):
        super().__init__(board, centers, freeCoefficients)

    def _calculateForPoint(self, point, distance):
        def distanceFunction(point, center):
            return distance(center, point) + self._freeCoefficients[center]

        return point, min(self._centers, key=functools.partial(distanceFunction, point))


class FuzzyPartitionWithFixedCentersAlgorithm(BasePartitionWithFixedCentersAlgorithm):

    def __init__(self, board, centers, stepFunction, confidenceDeegre, freeCoefficients, precision=0.01):
        super().__init__(board, centers, freeCoefficients)

        self._confidenceDeegre = confidenceDeegre
        self._precision        = precision
        self._stepFunction     = stepFunction

    def _calculateForPoint(self, point, distance):
        def calculateNextMembershipFunction(cell, previousMembershipFunction, helperFunction, center, freeCoeficient):
            key = -helperFunction / (2 * (distance(cell, center) + freeCoeficient))

            if  0 <= key <= 1:
                return key

            return 0.5 * (1 - sign(helperFunction + 2 * (previousMembershipFunction * distance(cell, center) + freeCoeficient)))

        membershipFunctions = [ 0 ] * len(self._centers)
        helperFunction      = 0

        gradient = 100

        self._stepFunction.reset()

        while not abs(gradient) < self._precision:
            gradient       = sum(membershipFunctions) - 1
            helperFunction = helperFunction + self._stepFunction.getNextStep() * gradient

            for i in range(0, len(membershipFunctions)):
                membershipFunctions[i] = calculateNextMembershipFunction(
                    point, membershipFunctions[i], helperFunction, self._centers[i], self._freeCoefficients[self._centers[i]])

        maximumMembershipFunction = max(membershipFunctions)
        center                    = self._centers[membershipFunctions.index(maximumMembershipFunction)]

        return point, center if maximumMembershipFunction > self._confidenceDeegre else None, membershipFunctions
