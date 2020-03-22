import abc
import functools
import copy
import concurrent.futures
import random

from .partition_utils import sign, LeftRectangularIntergate, RoundNextMultiple


WINDOW_SIZE = 500
cellSize    = 5


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


class BasePartitionWithNotFixedCentersAlgorithm(BasePartitionAlgorithm):

    def __init__(self, board, centersNumber, freeCoefficients):
        super().__init__()

        self._board            = board
        self._centersNumber    = centersNumber
        self._freeCoefficients = freeCoefficients


class FuzzyPartitionWithNotFixedCentersAlgorithm(BasePartitionWithNotFixedCentersAlgorithm):

    def __init__(self, board, centersNumber, stepFunction, confidenceDeegre, freeCoefficients, precision=0.01):
        super().__init__(board, centersNumber, freeCoefficients)

        self._confidenceDeegre = confidenceDeegre
        self._precision        = precision
        self._stepFunction     = stepFunction

    def calculatePartition(self, distance, startCenters):
        centers          = startCenters
        freeCoefficients = { center: 0 for center in centers }

        self._fuzzy_partition = FuzzyPartitionWithFixedCentersAlgorithm(
            self._board, centers, self._stepFunction, self._confidenceDeegre, freeCoefficients, self._precision)
        
        previousCenters   = centers
        newCenters, board = self._calculateForCenters(distance, previousCenters)
 
        yield board

        print('Old: ', centers)
        
        while previousCenters != newCenters:
            freeCoefficients = { center: 0 for center in newCenters }

            self._fuzzy_partition._freeCoefficients = freeCoefficients
            self._fuzzy_partition._centers          = newCenters

            previousCenters   = newCenters
            newCenters, board = self._calculateForCenters(distance, previousCenters)

            yield board

        print('New: ', newCenters)

    def _calculateForCenters(self, distance, centers):
        class Point:
            pass

        board = { key: Point() for key in self._board }

        for point, center, membershipFunctions in self._fuzzy_partition.calculatePartition(distance):
            board[point].center              = center
            board[point].membershipFunctions = membershipFunctions

        for center in centers:
            board[center].center              = center
            board[center].membershipFunctions = [center == item for item in centers]

        def grad_t_x_helper(x, y, distance, center):
            dis = distance((x, y), center)

            if dis == 0:
                return 0

            return (center[0] - x) / dis

        def grad_t_y_helper(x, y, distance, center):
            dis = distance((x, y), center)

            if dis == 0:
                return 0

            return (center[0] - y) / dis

        def function_grad_t_x(x, y, center, centers=centers, board=board, distance=distance):
            point = (x, y)

            return grad_t_x_helper(x, y, distance, center) * \
                board[point].membershipFunctions[centers.index(center)]

        def function_grad_t_y(x, y, center, centers=centers, board=board, distance=distance):
            point = (x, y)

            return grad_t_y_helper(x, y, distance, center) * \
                board[point].membershipFunctions[centers.index(center)]

        newFunction = functools.partial(
            self._costFunction, board=board, distance=distance, centers=centers)

        pointsNumber = int(WINDOW_SIZE / cellSize)
        integral     = LeftRectangularIntergate(
            newFunction, 0, WINDOW_SIZE, 0, WINDOW_SIZE, pointsNumber, pointsNumber)

        newCenters = centers[:]

        for center in centers:
            directionX = LeftRectangularIntergate(
                functools.partial(function_grad_t_x, center=center, centers=newCenters), 0, WINDOW_SIZE, 0, WINDOW_SIZE, pointsNumber, pointsNumber)

            directionY = LeftRectangularIntergate(
                functools.partial(function_grad_t_y, center=center, centers=newCenters), 0, WINDOW_SIZE, 0, WINDOW_SIZE, pointsNumber, pointsNumber)
        
            x = 10000
            y = 10000

            step = 1

            newFunction = functools.partial(
                self._costFunction, board=board, distance=distance)

            newIntergal = 1000000000

            print('Old Integral: ', integral)

            i = 0

            while newIntergal > integral:
                step /= 2

                x = center[0] - step * directionX
                y = center[1] - step * directionY

                i += 1

                def norm(n):
                    if n > WINDOW_SIZE:
                        return WINDOW_SIZE - cellSize
                    if n < 0:
                        return 0
                    return n

                x = norm(x)
                y = norm(y)

                newCenters[centers.index(center)] = (RoundNextMultiple(x), RoundNextMultiple(y))

                directionX = LeftRectangularIntergate(
                    functools.partial(function_grad_t_x, center=newCenters[centers.index(center)], centers=newCenters), 0, WINDOW_SIZE, 0, WINDOW_SIZE, pointsNumber, pointsNumber)

                directionY = LeftRectangularIntergate(
                    functools.partial(function_grad_t_y, center=newCenters[centers.index(center)], centers=newCenters), 0, WINDOW_SIZE, 0, WINDOW_SIZE, pointsNumber, pointsNumber)
        
                newIntergal = LeftRectangularIntergate(
                        functools.partial(newFunction, centers=newCenters),
                            0, WINDOW_SIZE, 0, WINDOW_SIZE, pointsNumber, pointsNumber)

            print('New Integral: ', newIntergal)

        return newCenters, board

    def _costFunction(self, x, y, board, distance, centers):
        point = (x, y)

        return sum(distance(center, point) * (membershipFunction ** 2) 
                for center, membershipFunction in zip(centers, board[point].membershipFunctions))

    