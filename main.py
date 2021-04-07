import math
import pygame
import numpy as np
from numpy.random import shuffle
from random import choice, randrange
from queue import PriorityQueue
from time import sleep

"""
Standard User Interface:
    Q key to quit
    P key to see alg / see alg step by step / hide alg
    L key to see maze gen / hide maze gen
Rest:
    O key to use diagonal conections or not
    R key to reset alg but keep board
    C key to reset entire board 
    SPACE key to start alg
    G key to generate maze
    F to loop maze generation and solving with animations
"""

pygame.init()

class Settings:

    # global settings variables
    def __init__(self):
        self.WIDTH = 1000
        self.HEIGHT = 500
        self.ROWS = 50
        self.COLS = 50
        self.SIZE = None # -----
        self.start_x = None # all 3 calculated in make_grid, start_x and start_y allows to center the grid on the screen
        self.start_y = None # -----
        self.NAME = "A* Path Finding"
        self.LOOP = False
        self.STEPS = 0 # 0 - see auto alg computation, 1 - force next algo step with RETURN, 2 - algo animation off
        self.GENERATE = False # False - maze generation without animation
        self.DIAGONAL = True # False - algo can't traverse on diagonals (diagonal move cost and normal move cost are equal)
        self.WIN = pygame.display.set_mode((self.WIDTH, self.HEIGHT)) # game window
        pygame.display.set_caption(self.NAME)

    def get_dims(self): # return grid dimentionr (rows x columns)
        return self.ROWS, self.COLS
    
    def draw(self, grid, reset=False, path=False): # draw all grid particles onto a screen
        self.WIN.fill(WHITE)

        for col in grid:
            for node in col:
                # keep just barrires, start and end nodes
                if reset and (node.is_closed() or node.is_open() or node.is_path()):
                    node.reset()
                
                # show just barriers and found path
                if path:
                    if node.is_closed() or node.is_open():
                        node.reset()
                node.draw()
        draw_grid() # draw lines and frame
        pygame.display.update()


# colors initialization
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165 ,0)
GREY = (128, 128, 128)

# settings initialization 
root = Settings()


class Node:
    """Graphical representation of graph node"""

    def __init__(self, col, row):
        self.row = row
        self.col = col
        self.x = root.start_x + col * root.SIZE # x position on the screeen
        self.y = root.start_y + row * root.SIZE # y position on the screen
        self.locked = False
        self.reset()

    def get_pos(self): # return position on the grid
        return self.row, self.col

    def is_blanc(self):
        return self.color == WHITE

    def is_closed(self):
        return self.color == RED

    def is_open(self):
        return self.color == GREEN

    def is_barrier(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == ORANGE

    def is_end(self):
        return self.color == PURPLE

    def is_path(self):
        return self.color == BLUE

    def reset(self): # reset all changed variables
        self.color = WHITE
        self.parent = None
        self.f = float("inf")
        self.g = float("inf")
        self.h = float("inf")

    def make_start(self):
        self.color = ORANGE

    def make_end(self):
        self.color = PURPLE

    def make_closed(self):
        self.color = RED

    def make_barrier(self):
        self.color = BLACK

    def make_path(self):
        self.color = BLUE

    def make_open(self):
        self.color = GREEN

    def draw(self): # draw node onto a screen
        pygame.draw.rect(root.WIN, self.color, (self.x, self.y, root.SIZE, root.SIZE))

    def _calculate_distances(self, end, start, g):
        """Args:
            f - total cost
            g - distance between current node and the start node
            h - distance between current node and the end node
        """
        # option with calculating every time g as euclidean distance as well
        # self.g = math.sqrt(math.pow((self.row - start.row), 2) + math.pow((self.col - start.col), 2))
        self.g = g
        self.h = math.sqrt(math.pow((self.row - end.row), 2) + math.pow((self.col - end.col), 2))
        f = g + self.h

        # if new cost is lower than previous this path is better
        if f < self.f:
            self.f = f
            return 1
        self.f = f
        return 0

    def _find_neighbors(self, grid, diagonal=True, 
    term=lambda col, row, grid: not grid[col][row].is_barrier()):
        """Args:
            grid - grid (type: list)
            diagonal - weather we can pick neighbors on diagonals (bool)
            term - validation term for neighbors
        returns list of all neighbors that fits the term
        """
        self.neighbors = []

        # calculate weather where we can expect neighbors (grid limits)
        down = self.row < root.ROWS - 1
        up = self.row > 0
        right = self.col < root.COLS - 1
        left = self.col > 0

        # DOWN
        if down and term(self.col, self.row + 1, grid):
            self.neighbors.append(grid[self.col][self.row + 1])
        # UP
        if up and term(self.col, self.row - 1, grid):
            self.neighbors.append(grid[self.col][self.row - 1])
        # RIGHT
        if right and term(self.col + 1, self.row, grid): 
            self.neighbors.append(grid[self.col + 1][self.row])
        # LEFT
        if left and term(self.col - 1, self.row, grid):
            self.neighbors.append(grid[self.col - 1][self.row])

        if diagonal: # take diagonal neighbors as well
            # LOWER RIGHT
            if up and right and term(self.col + 1, self.row + 1, grid):
                self.neighbors.append(grid[self.col + 1][self.row + 1])
            # LOWER LEFT
            if up and left and term(self.col - 1, self.row + 1, grid):
                self.neighbors.append(grid[self.col - 1][self.row + 1])
            # UPPER RIGHT
            if down and right and term(self.col + 1, self.row - 1, grid):
                self.neighbors.append(grid[self.col + 1][self.row - 1])
            # UPPER LEFT
            if down and left and term(self.col - 1, self.row - 1, grid):
                self.neighbors.append(grid[self.col - 1][self.row - 1])

    # to compare
    def __lt__(self, value):
        return False


class Maze:

    def __init__(self, grid):
        """Args:
            grid - full target grid (every node set as barrier)
        """
        self.H = len(grid)
        self.W = len(grid[0])
        self.h = int((self.H - 1) / 2)
        self.w = int((self.W - 1) / 2)
        self.grid = grid

    def genereate(self):
        # random initialization of first node
        crow = randrange(1, self.H, 2)
        ccol = randrange(1, self.W, 2)
        self.grid[crow][ccol].reset()
        num_visited = 1 # keep track of visited nodes

        while num_visited < self.h * self.w:
            for event in pygame.event.get(): # keep track of standard user interface
                listen(event)

            # find neighbors
            neighbors = self._find_neighbors(crow, ccol, self.grid, lambda x: x.is_barrier())

            # hif all neighbors were already visited, set random one as current and continue
            if len(neighbors) == 0:
                (crow, ccol) = choice(self._find_neighbors(crow, ccol, self.grid, lambda x: x.is_blanc()))
                continue

            # loop through all found neighbors
            for nrow, ncol in neighbors:
                if self.grid[nrow][ncol].is_barrier(): # if neighbor wasn't already traversed
                    self.grid[(nrow + crow) // 2][(ncol + ccol) // 2].reset() # open wall toward it
                    self.grid[nrow][ncol].reset() # set it as visited
                    num_visited += 1 # update visited count

                    # current becomes new neighbor
                    crow = nrow
                    ccol = ncol

                    break

            if root.GENERATE: # visualizate every iteration onto screen
                root.draw(self.grid)

        return self.grid

    def _find_neighbors(self, r, c, grid, term):
        """Args:
            r (int): row of cell of interest
            c (int): column of cell of interest
            grid (np.array): 2D maze grid
            is_wall (bool): Are we looking for neighbors that are walls, or open cells?
        """
        n = []

        if r > 1 and term(grid[r - 2][c]):
            n.append((r - 2, c))
        if r < self.H - 2 and term(grid[r + 2][c]):
            n.append((r + 2, c))
        if c > 1 and term(grid[r][c - 2]):
            n.append((r, c - 2))
        if c < self.W - 2 and term(grid[r][c + 2]):
            n.append((r, c + 2))
        
        shuffle(n)
        return n


def genereate_maze(): # generate the maze with size definied with S(ettings.COLS, Settings.ROWS)
    # Make sure both dimentions are odd
    if root.ROWS % 2 != 1:
        root.ROWS += 1
    if root.COLS % 2 != 1:
        root.COLS += 1

    # generate full grid
    grid = make_grid(True)

    # generate the maze
    m = Maze(grid)
    grid = m.genereate()

    return grid

def make_grid(full=False):
    """Args:
        full - True - all nodes are berriers, False - all nodes are traversable
    """
    grid = []
    # adjust tiles size to smaller dimention
    root.SIZE = min(root.HEIGHT / root.ROWS, root.WIDTH / root.COLS)
    root.start_x = (root.WIDTH - root.COLS * root.SIZE) / 2
    root.start_y = (root.HEIGHT - root.ROWS * root.SIZE) / 2

    # fill the grid with empty nodes
    for i in range(root.COLS):
        grid.append([])
        for j in range(root.ROWS):
            node = Node(i, j)

            if full:
                node.make_barrier()

            grid[i].append(node)

    # make a frame
    for index, row in enumerate(grid):
        if index == 0 or index == len(grid) - 1:
            for node in row:
                node.make_barrier()
                node.locked = True
        else:
            row[0].make_barrier()
            row[0].locked = True
            row[-1].make_barrier()
            row[-1].locked = True

    return grid

def draw_grid(): # draw grid lines onto a screen
    for i in range(root.ROWS + 1):
        pygame.draw.line(root.WIN, GREY, (root.start_x, root.start_y + i * root.SIZE), (root.start_x + root.COLS * root.SIZE, root.start_y + i * root.SIZE))
    for j in range(root.COLS + 1):
        pygame.draw.line(root.WIN, GREY, (root.start_x + j * root.SIZE, root.start_y), (root.start_x + j * root.SIZE, root.start_y + root.ROWS * root.SIZE))

def get_clicked_pos(pos): # returns mouse pos (row, col) if it collides with the grid else (-1, None)
    """Args:
        pos - pygame.mouse.get_pos()
    """
    x, y  = pos

    # if mouse pos is on the grid
    if root.start_x < x < root.WIDTH - root.start_x and root.start_y < y < root.HEIGHT - root.start_y:
        row = (y - root.start_y )// root.SIZE
        col = (x - root.start_x) // root.SIZE
        return int(row), int(col)
    
    return -1, None

def listen(event): # user interaction available all the time - SUI 
    """Args:
        event - object from pygame.event.get()
    """
    if event.type == pygame.QUIT: # quit
        pygame.quit()
        exit()
    elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_p: # change alg display mode
            root.STEPS = (root.STEPS + 1) % 3
        elif event.key == pygame.K_q: # quit
            pygame.quit()
            exit()
        elif event.key == pygame.K_l: # change maze generator display mode
            root.GENERATE = not root.GENERATE
        elif event.key == pygame.K_f: # loop maze generator and solver
            root.LOOP = not root.LOOP

def algorithm(start, end, grid):
    """Args:
        start - start node instance
        end - end node instance
        grid - grid (type: list)
    """
    open_set = PriorityQueue() # queue initialization

    # calculate start node distances and add it to open_set
    start._calculate_distances(end, start, 0)
    open_set.put((0, 0, start))

    while not open_set.empty():
        for event in pygame.event.get(): # keep track of standard user interface
            listen(event)

        current = open_set.get()[2] # index 2 is node

        if current == end: # we found the shortest path
            cost = draw_path(current.parent, start) # calculate cost (number of traversed tiles)
            if root.STEPS == 2: # draw just a path (Node.is_path()) on the grid
                root.draw(grid, path=True)
            else:
                root.draw(grid) # draw all types of Nodes
            #return end.g
            return cost

        if current.is_closed():
            # node was already visited (case when during
            # distance updates we found better and added it into
            # open_set without removing worse option (more expensive in 
            # CPU than this solution))
            continue

        if current != start:
            # keep start node ORANGE
            current.make_closed()

        current._find_neighbors(grid, diagonal=root.DIAGONAL) # find all no-barrier neighbors

        for neighbor in current.neighbors:
            neighbor._calculate_distances(end, start, current.g + 1)

            if neighbor.is_closed():
                # neighbor was already visited
                continue

            if not neighbor.is_open():
                neighbor.parent = current # this path was shorter or it's a first path - update node's parent

                if neighbor != start and neighbor != end and not neighbor.is_open():
                    # this node haven't been visited yet, mark it as open
                    neighbor.make_open()

                # add best configuration to open_set
                open_set.put((neighbor.f, neighbor.h, neighbor))
 
        if root.STEPS != 2: # draw all nodes onto a screen
            root.draw(grid)

        if root.STEPS == 1: # to control every iteration of alhgorithm - wait until RETURN pressed
            run = 0
            while not run and root.STEPS == 1:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RETURN:
                            run = 1

                    listen(event) # keep track of standard user interface

    # path doesn't exist
    return -1   

def draw_path(node, start):
    # mark path of best solution and calculate its distance (cost)
    cost = 0
    while node != start:
        node.make_path()
        cost += 1
        node = node.parent
    cost += 1
    return cost

def get_node(pos, grid):
    """Args:
        pos - pygame.mouse.get_pos()
    """
    row, col = get_clicked_pos(pos)

    if row == -1: # mouse haven't colide the grid area
        return 0
    return grid[col][row] # node - mouse target

def genereate_full_maze(): # generate maze with start and end
    start, end = None, None
    grid = genereate_maze()
    start = grid[1][1]
    end = grid[-2][-2]
    start.make_start()
    end.make_end()

    return start, end, grid

def solve(start, end, grid):
    solution = algorithm(start, end, grid)
    if solution != -1:
        print(f"Shortest path is {solution} units long")
    else:
        print("Path doesn't exist")

def main(): # main program function
    grid = make_grid() # create an empty grid

    start, end = None, None # initializa start and end nodes

    started = False
    while 1:
        root.draw(grid) # update the screen

        for event in pygame.event.get():
            listen(event) # keep track of standard user interface

            if started: 
                continue

            # LEFT 
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                node = get_node(pos, grid)

                if not node:
                    continue
                
                if not start and node != end and not node.locked: # make it a start node
                    start = node
                    start.make_start()

                elif not end and node != start and not node.locked: # make it an end node
                    end = node
                    end.make_end()
                
                elif node != end and node != start: # make it barrier
                    node.make_barrier()

            # RIGHT
            elif pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                node = get_node(pos, grid)

                if not node:
                    continue

                elif not node.locked: # if it isn't part of the frame
                    node.reset()

                    if node == start:
                        start = None
                    elif node == end:
                        end = None

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end: # start the alg
                    solve(start, end, grid)

                if event.key == pygame.K_c: # clear the board
                    start, end = None, None
                    grid = make_grid()

                elif event.key == pygame.K_r: # clear the board but keep barriers, start and end nodes
                    root.draw(grid, True)

                elif event.key == pygame.K_g: # generate the maze
                    if root.LOOP:
                        root.DIAGONAL = False
                        root.GENERATE = True
                        root.STEPS = 0
                        while root.LOOP:
                            for event in pygame.event.get():
                                listen(event)
                            start, end, grid = genereate_full_maze()
                            solve(start, end, grid)
                            sleep(3)
                    else:
                        start, end, grid = genereate_full_maze()

                elif event.key == pygame.K_o: # change diagonal setting
                    root.DIAGONAL = not root.DIAGONAL

main()
