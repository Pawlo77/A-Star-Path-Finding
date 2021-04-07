# a_star_path_finding
Graphical implementation of A* path finding algorithm using python and pygame. 
It has random generator of perfect mazes. 

Options (key map):
Standard User Interface - instant response:
    Q key to quit
    P key to see alg / see alg step by step / hide alg
    L key to see maze gen / hide maze gen
Rest - response only in graph creation mode 
(not available during alg work):
    O key to use diagonal conections or not
    R key to reset alg but keep board
    C key to reset entire board 
    SPACE key to start alg
    G key to generate maze
    F to loop maze generation and solving with animations

To change graph size:
  change intiger values of rows (Settings().ROWS) or columns (Settings().COLS)
To change window size:
  change intiger values of x window dimension  (Settings().WIDTH)
  or y window dimension  (Settings().HEIGHT)
