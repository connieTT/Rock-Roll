# Author: Tongtong Zhao/ Saozhong Han
# Robot stops when obstacle encoutered
# Robot follows wall
from myro import *
init("/dev/tty.Fluke2-0B55-Fluke2")

DEFUALT_OBSTACLE = 6000
RIGHT_CHECK = 1
LEFT_CHECK = 1	

while 1:
	if (getObstacle(1)<DEFUALT_OBSTACLE):
		forward(1)
	else:
		stop()
		break

turnBy(-90,"deg")

while 1:
	if (LEFT_CHECK==0):
		stop()
		break
	else:
		forward(1,1)
		turnBy(90,"deg")
		
		if (getObstacle(1)>=DEFUALT_OBSTACLE):
			LEFT_CHECK = 1
			print(getObstacle(1))
		else:
			LEFT_CHECK = 0
			print(getObstacle(1))
		
		turnBy(-90,"deg")

for x in range(2):
	y = [(500, .75), (800, .5), (900, .75), (880, .75), (700,1)]
	playSong(y)
	speak("Yeah Yeah We did it.")