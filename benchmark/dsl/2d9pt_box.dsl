def stencil_2d9pt_star(lattice:[4096,4098,'float']):
  for y in range(1,4095):
    for x in range(1,4097):
      lattice[t+1,y,x]=0.1*lattice[t,y-1,x-1]+0.3*lattice[t,y-1,x+1]+0.2*lattice[t,y-1,x]+0.4*lattice[t,y,x-1]+0.5*lattice[t,y,x]+0.6*lattice[t,y,x+1]+0.7*lattice[t,y+1,x-1]+0.8*lattice[t,y+1,x]+0.9*lattice[t,y+1,x+1]