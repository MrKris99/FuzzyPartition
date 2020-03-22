import NonLinearOptimization

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


def function(t):
	x, y = t

	def f(x, y):
		return x ** 2 + y ** 2
	return LeftRectangularIntergate(f, 0, 100, 0, 100, 400, 400)


r = NonLinearOptimization.r_algorithm(function, [-10, -10])
print(r[-1])
