Q-learning-in-path-planning
=======
1.Environment 
------------
exceed python 3.0 version

2.dependency
-------------------
pygame <br>
sys <br>
random <br>
math <br>
numpy<br>
time <br>

3.version
--------------
(1)Q-learning-v1.0:<br>
  this version in path planning will  not pass through any obstacles<br>
(2)Q-learning-v1.1:<br>
  this version in path planning will pass through obstacles in Initial stage

4.parameter
----------------------
  `XDIM YDIM`: The length and width of the environment map<br>
  `epochs`: Total training times<br>
  `GAME_LEVEL`: Select the environment map,you can create environment by yourself<br>
  `gamma`: learning rate<br>
  `loss`: loss rate<br>
  `Pmin`: Minimum forward unit<br>
  
5.Operation method
-----------------------
  1.If there is no dependency problem, you can run the code directly, and the environment map will be displayed.<br>
  2.Click the left key of the mouse to select the starting point.<br>
  3.Then click the left click of the mouse again to select the target point.<br>
  4.The code will start planning the path.<br>
  
6.training method
----------------------------
The gravitational potential field is added at the beginning stage to speed up the training speed.<br>
Then use standard Q-learning to explore the environment.<br>

PS: Code writing is not standard, please forgive me
---------------------------------
