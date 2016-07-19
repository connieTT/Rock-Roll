# Author: Tongtong Zhao & Saozhong Han
# Lab2

from collections import defaultdict
from myro import *

import csv
import math

def parseGraph(filePath):
    # Return adjacency list and position list parsed from an input file.
    with open(filePath, 'r') as cvsfile:
        reader = csv.reader(cvsfile, delimiter=',')

        adjacentList = defaultdict(list)
        positionList = {}
        for row in reader:
            if (len(row) == 3):
                positionList[row[0]] = (row[1], row[2])
            if (len(row) == 2):
                adjacentList[row[0]].append(row[1])
                adjacentList[row[1]].append(row[0])

        return (adjacentList, positionList)

def selectNode(visited, start, targets, adjacentList):
    # Selects the node with the most edges
    for i,j in visited:
        if i == start and j in targets: targets.remove(j)

    number = lambda x: len(adjacentList[x])

    targets = list(sorted(targets, key=number, reverse=True))
    if not targets: return None

    node = targets[0]
    
    return node


def selectOdd(visited, start, targets, adjacentList):
    # Selects the node with the most odd edges
    for i,j in visited:
        if i == start and j in targets: targets.remove(j)

    number = lambda x: len(adjacentList[x])

    targets = list(sorted(targets, key=number, reverse=True))
    if not targets: return None

    node = targets[0]
    
    # prefer the highest odd edges if it exits
    for n in targets[1:]:
        if number(n) % 2 == 1:
            node = n
            break

    return node

def findPath(adjacentList, positionList):
    visited = []
    
    current = selectOdd(visited, None, list(adjacentList.keys()), adjacentList)

    path = [current]
    done = False
    while not done:
        n = selectNode(visited, current, adjacentList[current], adjacentList)       
        visited.append((current, n))
        visited.append((n, current))

        if not n: done = True
        else:
            path.append(n)

        current = n

    return path

def goPath(path, positionList):
    # trajectory is a list of angles and distances
    trajectory = []
    currentAngle = 0
    for current, n in zip(path[:-1], path[1:]):
        deltaX = float(positionList[n][0]) - float(positionList[current][0])
        deltaY = float(positionList[n][1]) - float(positionList[current][1])

        distance = math.hypot(deltaX, deltaY)
        angle = math.atan2(deltaY, deltaX) - currentAngle

        trajectory.append((distance, angle))
        currentAngle += angle

    for distance, angle in trajectory:
        turnBy(int(round(math.degrees(angle))), 'deg')
        forward(1, distance)

def main():
    filePath = 'CS3630_Lab2_Map3.csv'

    adjacentList, positionList = parseGraph(filePath)
    path = findPath(adjacentList, positionList)
    print(', '.join(path))
    print (positionList)

    init("/dev/tty.Fluke2-0B55-Fluke2")
    goPath(path, positionList)


main()

